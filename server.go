// Command nogodsnomasters serves the published compiler-inspection site and
// turns trusted local Python uploads into versioned Turing page bundles.
package main

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"io"
	"log"
	"mime"
	"net"
	"net/http"
	"net/url"
	"os"
	"os/exec"
	"path/filepath"
	"sort"
	"strings"
	"sync"
	"time"
)

const (
	gallerySchema  = "turing-gallery-v1"
	defaultAddress = "127.0.0.1:8787"
	defaultMaxFile = int64(2 << 20)
)

type configuration struct {
	root       string
	turing     string
	python     string
	maxUpload  int64
	buildLimit time.Duration
}

type siteServer struct {
	config     configuration
	static     http.Handler
	generation sync.Mutex
}

type artifactRecord struct {
	Path  string `json:"path"`
	Bytes int64  `json:"bytes"`
}

type bundleManifest struct {
	Schema  string `json:"schema"`
	Program struct {
		Slug       string `json:"slug"`
		Title      string `json:"title"`
		Entrypoint string `json:"entrypoint"`
	} `json:"program"`
	Version struct {
		ID string `json:"id"`
	} `json:"version"`
	CreatedAt string `json:"created_at"`
	Page      struct {
		Path string `json:"path"`
	} `json:"page"`
	Source struct {
		Filename string `json:"filename"`
	} `json:"source"`
	Artifacts []artifactRecord `json:"artifacts"`
}

type galleryItem struct {
	Slug       string `json:"slug"`
	Title      string `json:"title"`
	Entrypoint string `json:"entrypoint"`
	Version    string `json:"version"`
	CreatedAt  string `json:"created_at"`
	URL        string `json:"url"`
	Source     string `json:"source"`
	Artifacts  int    `json:"artifacts"`
	Bytes      int64  `json:"bytes"`
	Latest     bool   `json:"latest"`
}

type commandResult struct {
	OK       bool                   `json:"ok"`
	URL      string                 `json:"url"`
	Bundle   string                 `json:"bundle"`
	Page     string                 `json:"page"`
	Manifest map[string]interface{} `json:"manifest"`
}

type callableRequest struct {
	Source    string                 `json:"source"`
	Callable  string                 `json:"callable"`
	Arguments map[string]interface{} `json:"arguments"`
}

func newSiteServer(config configuration) (*siteServer, error) {
	root, err := filepath.Abs(config.root)
	if err != nil {
		return nil, fmt.Errorf("resolve site root: %w", err)
	}
	turing, err := filepath.Abs(config.turing)
	if err != nil {
		return nil, fmt.Errorf("resolve Turing root: %w", err)
	}
	config.root = root
	config.turing = turing
	if config.maxUpload <= 0 {
		return nil, errors.New("max upload must be positive")
	}
	if config.buildLimit <= 0 {
		return nil, errors.New("build timeout must be positive")
	}
	return &siteServer{config: config, static: http.FileServer(http.Dir(root))}, nil
}

func (server *siteServer) handler() http.Handler {
	mux := http.NewServeMux()
	mux.HandleFunc("/api/health", server.health)
	mux.HandleFunc("/api/gallery", server.gallery)
	mux.HandleFunc("/api/generate", server.generate)
	mux.HandleFunc("/api/run", server.runCallable)
	mux.HandleFunc("/", server.serveStatic)
	return http.HandlerFunc(func(writer http.ResponseWriter, request *http.Request) {
		writer.Header().Set("X-Content-Type-Options", "nosniff")
		mux.ServeHTTP(writer, request)
	})
}

func (server *siteServer) runCallable(writer http.ResponseWriter, request *http.Request) {
	if request.Method != http.MethodPost {
		methodNotAllowed(writer, http.MethodPost)
		return
	}
	if !loopbackRemote(request.RemoteAddr) || !localOrigin(request) {
		writeJSON(writer, http.StatusForbidden, map[string]string{"error": "callable execution is restricted to loopback clients and origins"})
		return
	}
	request.Body = http.MaxBytesReader(writer, request.Body, 1<<20)
	var invocation callableRequest
	if err := json.NewDecoder(request.Body).Decode(&invocation); err != nil {
		writeJSON(writer, http.StatusBadRequest, map[string]string{"error": "invalid callable request: " + err.Error()})
		return
	}
	if invocation.Callable == "" || strings.ContainsAny(invocation.Callable, ".\\/:") {
		writeJSON(writer, http.StatusBadRequest, map[string]string{"error": "callable must name one module-level function"})
		return
	}
	cleanURL := filepath.ToSlash(filepath.Clean("/" + strings.TrimPrefix(invocation.Source, "/")))
	if !strings.HasPrefix(cleanURL, "/site/programs/") || !strings.Contains(cleanURL, "/source/python_source/") || !strings.EqualFold(filepath.Ext(cleanURL), ".py") {
		writeJSON(writer, http.StatusBadRequest, map[string]string{"error": "source must be a published Python bundle file"})
		return
	}
	sourcePath := filepath.Clean(filepath.Join(server.config.root, filepath.FromSlash(strings.TrimPrefix(cleanURL, "/"))))
	relative, err := filepath.Rel(server.config.root, sourcePath)
	if err != nil || relative == ".." || strings.HasPrefix(relative, ".."+string(filepath.Separator)) {
		writeJSON(writer, http.StatusBadRequest, map[string]string{"error": "source escapes the published site root"})
		return
	}
	if info, err := os.Stat(sourcePath); err != nil || info.IsDir() {
		writeJSON(writer, http.StatusNotFound, map[string]string{"error": "published source not found"})
		return
	}
	encodedArguments, err := json.Marshal(invocation.Arguments)
	if err != nil {
		writeJSON(writer, http.StatusBadRequest, map[string]string{"error": "arguments are not JSON serializable"})
		return
	}
	ctx, cancel := context.WithTimeout(request.Context(), server.config.buildLimit)
	defer cancel()
	command := exec.CommandContext(
		ctx,
		server.config.python,
		filepath.Join(server.config.turing, "run_site_callable.py"),
		"--source", sourcePath,
		"--callable", invocation.Callable,
		"--arguments-json", string(encodedArguments),
	)
	command.Dir = server.config.turing
	command.Env = append(os.Environ(), "PYGAME_HIDE_SUPPORT_PROMPT=1")
	var output cappedBuffer
	output.limit = 16 << 20
	command.Stdout = &output
	command.Stderr = &output
	if err := command.Run(); err != nil {
		message := strings.TrimSpace(output.String())
		if message == "" {
			message = err.Error()
		}
		writeJSON(writer, http.StatusUnprocessableEntity, map[string]string{"error": message})
		return
	}
	writer.Header().Set("Content-Type", "application/json")
	writer.Header().Set("Cache-Control", "no-store")
	writer.WriteHeader(http.StatusOK)
	_, _ = writer.Write(output.Bytes())
}

func (server *siteServer) health(writer http.ResponseWriter, request *http.Request) {
	if request.Method != http.MethodGet {
		methodNotAllowed(writer, http.MethodGet)
		return
	}
	writeJSON(writer, http.StatusOK, map[string]interface{}{
		"ok": true, "service": "nogodsnomasters", "gallery_schema": gallerySchema,
	})
}

func (server *siteServer) gallery(writer http.ResponseWriter, request *http.Request) {
	if request.Method != http.MethodGet {
		methodNotAllowed(writer, http.MethodGet)
		return
	}
	if !localOrigin(request) {
		writeJSON(writer, http.StatusForbidden, map[string]string{"error": "gallery API accepts only loopback origins"})
		return
	}
	items, err := server.discoverGallery()
	if err != nil {
		writeJSON(writer, http.StatusInternalServerError, map[string]string{"error": err.Error()})
		return
	}
	writer.Header().Set("Cache-Control", "no-store")
	writeJSON(writer, http.StatusOK, map[string]interface{}{
		"schema": gallerySchema,
		"root":   "/site/programs/",
		"items":  items,
	})
}

func (server *siteServer) discoverGallery() ([]galleryItem, error) {
	pattern := filepath.Join(server.config.root, "site", "programs", "*", "versions", "*", "bundle.json")
	manifests, err := filepath.Glob(pattern)
	if err != nil {
		return nil, fmt.Errorf("scan bundle tree: %w", err)
	}
	items := make([]galleryItem, 0, len(manifests))
	for _, manifestPath := range manifests {
		body, err := os.ReadFile(manifestPath)
		if err != nil {
			continue
		}
		var manifest bundleManifest
		if json.Unmarshal(body, &manifest) != nil || manifest.Schema != "turing-program-bundle-v1" {
			continue
		}
		if manifest.Program.Slug == "" || manifest.Version.ID == "" || manifest.Page.Path == "" {
			continue
		}
		bundleDirectory := filepath.Dir(manifestPath)
		pagePath := filepath.Clean(filepath.Join(bundleDirectory, filepath.FromSlash(manifest.Page.Path)))
		relativePage, err := filepath.Rel(server.config.root, pagePath)
		if err != nil || strings.HasPrefix(relativePage, ".."+string(filepath.Separator)) || relativePage == ".." {
			continue
		}
		if info, err := os.Stat(pagePath); err != nil || info.IsDir() {
			continue
		}
		var total int64
		for _, artifact := range manifest.Artifacts {
			total += artifact.Bytes
		}
		items = append(items, galleryItem{
			Slug: manifest.Program.Slug, Title: manifest.Program.Title,
			Entrypoint: manifest.Program.Entrypoint, Version: manifest.Version.ID,
			CreatedAt: manifest.CreatedAt, URL: "/" + filepath.ToSlash(relativePage),
			Source: manifest.Source.Filename, Artifacts: len(manifest.Artifacts), Bytes: total,
		})
	}
	sort.Slice(items, func(left, right int) bool {
		if items[left].CreatedAt != items[right].CreatedAt {
			return items[left].CreatedAt > items[right].CreatedAt
		}
		if items[left].Slug != items[right].Slug {
			return items[left].Slug < items[right].Slug
		}
		return items[left].Version > items[right].Version
	})
	seen := make(map[string]bool)
	for index := range items {
		if !seen[items[index].Slug] {
			items[index].Latest = true
			seen[items[index].Slug] = true
		}
	}
	return items, nil
}

func (server *siteServer) generate(writer http.ResponseWriter, request *http.Request) {
	if request.Method != http.MethodPost {
		methodNotAllowed(writer, http.MethodPost)
		return
	}
	if !loopbackRemote(request.RemoteAddr) || !localOrigin(request) {
		writeJSON(writer, http.StatusForbidden, map[string]string{"error": "generation is restricted to loopback clients and origins"})
		return
	}
	request.Body = http.MaxBytesReader(writer, request.Body, server.config.maxUpload+(1<<20))
	if err := request.ParseMultipartForm(1 << 20); err != nil {
		writeJSON(writer, http.StatusBadRequest, map[string]string{"error": "invalid multipart upload: " + err.Error()})
		return
	}
	file, header, err := request.FormFile("source")
	if err != nil {
		writeJSON(writer, http.StatusBadRequest, map[string]string{"error": "missing Python file field 'source'"})
		return
	}
	defer file.Close()
	filename := filepath.Base(header.Filename)
	if !strings.EqualFold(filepath.Ext(filename), ".py") {
		writeJSON(writer, http.StatusBadRequest, map[string]string{"error": "source filename must end in .py"})
		return
	}
	body, err := io.ReadAll(io.LimitReader(file, server.config.maxUpload+1))
	if err != nil || int64(len(body)) > server.config.maxUpload {
		writeJSON(writer, http.StatusRequestEntityTooLarge, map[string]string{"error": "Python source exceeds upload limit"})
		return
	}
	probes := strings.TrimSpace(request.FormValue("probes"))
	if probes == "" {
		probes = "{}"
	}
	var probeObject map[string]interface{}
	if json.Unmarshal([]byte(probes), &probeObject) != nil {
		writeJSON(writer, http.StatusBadRequest, map[string]string{"error": "probe values must be a JSON object"})
		return
	}

	temporary, err := os.MkdirTemp("", "nogodsnomasters-upload-")
	if err != nil {
		writeJSON(writer, http.StatusInternalServerError, map[string]string{"error": "create upload workspace: " + err.Error()})
		return
	}
	defer os.RemoveAll(temporary)
	sourcePath := filepath.Join(temporary, filename)
	if err := os.WriteFile(sourcePath, body, 0600); err != nil {
		writeJSON(writer, http.StatusInternalServerError, map[string]string{"error": "store upload: " + err.Error()})
		return
	}
	resultPath := filepath.Join(temporary, "result.json")
	arguments := []string{
		filepath.Join(server.config.turing, "build_site_page.py"),
		"--source", sourcePath,
		"--destination", server.config.root,
		"--probes-json", probes,
		"--result-json", resultPath,
	}
	for _, field := range []string{"entrypoint", "title", "slug"} {
		value := strings.TrimSpace(request.FormValue(field))
		if len(value) > 200 {
			writeJSON(writer, http.StatusBadRequest, map[string]string{"error": field + " is too long"})
			return
		}
		if value != "" {
			arguments = append(arguments, "--"+field, value)
		}
	}

	server.generation.Lock()
	defer server.generation.Unlock()
	ctx, cancel := context.WithTimeout(request.Context(), server.config.buildLimit)
	defer cancel()
	command := exec.CommandContext(ctx, server.config.python, arguments...)
	command.Dir = server.config.turing
	command.Env = append(os.Environ(), "PYGAME_HIDE_SUPPORT_PROMPT=1")
	var output cappedBuffer
	output.limit = 128 << 10
	command.Stdout = &output
	command.Stderr = &output
	err = command.Run()
	if ctx.Err() == context.DeadlineExceeded {
		writeJSON(writer, http.StatusGatewayTimeout, map[string]string{"error": "page generation exceeded " + server.config.buildLimit.String()})
		return
	}
	if err != nil {
		message := strings.TrimSpace(output.String())
		if message == "" {
			message = err.Error()
		}
		writeJSON(writer, http.StatusUnprocessableEntity, map[string]string{"error": message})
		return
	}
	resultBody, err := os.ReadFile(resultPath)
	if err != nil {
		writeJSON(writer, http.StatusInternalServerError, map[string]string{"error": "builder did not return result metadata"})
		return
	}
	var result commandResult
	if json.Unmarshal(resultBody, &result) != nil || !result.OK || !strings.HasPrefix(result.URL, "/site/programs/") {
		writeJSON(writer, http.StatusInternalServerError, map[string]string{"error": "builder returned an invalid bundle result"})
		return
	}
	writer.Header().Set("Cache-Control", "no-store")
	writeJSON(writer, http.StatusCreated, result)
}

func (server *siteServer) serveStatic(writer http.ResponseWriter, request *http.Request) {
	if request.Method != http.MethodGet && request.Method != http.MethodHead {
		methodNotAllowed(writer, http.MethodGet, http.MethodHead)
		return
	}
	if request.URL.Path == "/" {
		writer.Header().Set("Cache-Control", "no-cache")
	} else if strings.Contains(request.URL.Path, "/versions/") {
		writer.Header().Set("Cache-Control", "public, max-age=31536000, immutable")
	}
	server.static.ServeHTTP(writer, request)
}

type cappedBuffer struct {
	bytes.Buffer
	limit int
}

func (buffer *cappedBuffer) Write(body []byte) (int, error) {
	original := len(body)
	remaining := buffer.limit - buffer.Len()
	if remaining > 0 {
		if len(body) > remaining {
			body = body[:remaining]
		}
		_, _ = buffer.Buffer.Write(body)
	}
	return original, nil
}

func loopbackRemote(remote string) bool {
	host, _, err := net.SplitHostPort(remote)
	if err != nil {
		host = remote
	}
	if strings.EqualFold(host, "localhost") {
		return true
	}
	ip := net.ParseIP(strings.Trim(host, "[]"))
	return ip != nil && ip.IsLoopback()
}

func localOrigin(request *http.Request) bool {
	origin := request.Header.Get("Origin")
	if origin == "" {
		return true
	}
	parsed, err := url.Parse(origin)
	if err != nil {
		return false
	}
	host := parsed.Hostname()
	if strings.EqualFold(host, "localhost") {
		return true
	}
	ip := net.ParseIP(host)
	return ip != nil && ip.IsLoopback()
}

func methodNotAllowed(writer http.ResponseWriter, methods ...string) {
	writer.Header().Set("Allow", strings.Join(methods, ", "))
	writeJSON(writer, http.StatusMethodNotAllowed, map[string]string{"error": "method not allowed"})
}

func writeJSON(writer http.ResponseWriter, status int, value interface{}) {
	writer.Header().Set("Content-Type", "application/json; charset=utf-8")
	writer.WriteHeader(status)
	_ = json.NewEncoder(writer).Encode(value)
}

func main() {
	address := flag.String("addr", defaultAddress, "loopback listen address")
	root := flag.String("root", ".", "published site root")
	turing := flag.String("turing", "./turing", "Turing repository root")
	python := flag.String("python", "python", "Python interpreter used by the page builder")
	maxUpload := flag.Int64("max-upload", defaultMaxFile, "maximum Python source bytes")
	timeout := flag.Duration("build-timeout", 10*time.Minute, "maximum compiler run time")
	flag.Parse()

	host, _, err := net.SplitHostPort(*address)
	if err != nil {
		log.Fatalf("invalid -addr: %v", err)
	}
	if ip := net.ParseIP(host); !strings.EqualFold(host, "localhost") && (ip == nil || !ip.IsLoopback()) {
		log.Fatalf("refusing non-loopback listen address %q", *address)
	}
	_ = mime.AddExtensionType(".wasm", "application/wasm")
	server, err := newSiteServer(configuration{
		root: *root, turing: *turing, python: *python,
		maxUpload: *maxUpload, buildLimit: *timeout,
	})
	if err != nil {
		log.Fatal(err)
	}
	log.Printf("serving %s at http://localhost:%s", server.config.root, strings.Split(*address, ":")[1])
	log.Printf("trusted Python publisher: POST /api/generate (Turing: %s)", server.config.turing)
	if err := http.ListenAndServe(*address, server.handler()); err != nil {
		log.Fatal(err)
	}
}
