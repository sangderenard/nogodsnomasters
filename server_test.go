package main

import (
	"bytes"
	"encoding/json"
	"mime/multipart"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"testing"
	"time"
)

func testServer(t *testing.T) (*siteServer, string) {
	t.Helper()
	root := t.TempDir()
	turing := t.TempDir()
	if err := os.WriteFile(filepath.Join(root, "index.html"), []byte("homepage"), 0600); err != nil {
		t.Fatal(err)
	}
	server, err := newSiteServer(configuration{
		root: root, turing: turing, python: "python",
		maxUpload: 1 << 20, buildLimit: time.Minute,
	})
	if err != nil {
		t.Fatal(err)
	}
	return server, root
}

func writeBundle(t *testing.T, root, slug, version, created string) {
	t.Helper()
	directory := filepath.Join(root, "site", "programs", slug, "versions", version)
	if err := os.MkdirAll(directory, 0700); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(directory, "index.html"), []byte(slug), 0600); err != nil {
		t.Fatal(err)
	}
	manifest := map[string]interface{}{
		"schema":     "turing-program-bundle-v1",
		"program":    map[string]string{"slug": slug, "title": "Title " + slug, "entrypoint": "kernel"},
		"version":    map[string]string{"id": version},
		"created_at": created,
		"page":       map[string]string{"path": "index.html"},
		"source":     map[string]string{"filename": slug + ".py"},
		"artifacts":  []map[string]interface{}{{"path": "index.html", "bytes": 7}},
	}
	body, _ := json.Marshal(manifest)
	if err := os.WriteFile(filepath.Join(directory, "bundle.json"), body, 0600); err != nil {
		t.Fatal(err)
	}
}

func TestGalleryIsInferredFromVersionedBundleTree(t *testing.T) {
	server, root := testServer(t)
	writeBundle(t, root, "alpha", "v1-old", "2026-01-01T00:00:00Z")
	writeBundle(t, root, "alpha", "v1-new", "2026-02-01T00:00:00Z")
	writeBundle(t, root, "beta", "v1-only", "2026-01-15T00:00:00Z")

	request := httptest.NewRequest(http.MethodGet, "/api/gallery", nil)
	request.RemoteAddr = "127.0.0.1:1234"
	response := httptest.NewRecorder()
	server.handler().ServeHTTP(response, request)

	if response.Code != http.StatusOK {
		t.Fatalf("status %d: %s", response.Code, response.Body.String())
	}
	var payload struct {
		Schema string        `json:"schema"`
		Items  []galleryItem `json:"items"`
	}
	if err := json.Unmarshal(response.Body.Bytes(), &payload); err != nil {
		t.Fatal(err)
	}
	if payload.Schema != gallerySchema || len(payload.Items) != 3 {
		t.Fatalf("unexpected gallery: %#v", payload)
	}
	if payload.Items[0].Version != "v1-new" || !payload.Items[0].Latest {
		t.Fatalf("newest alpha version not first/latest: %#v", payload.Items[0])
	}
	if payload.Items[0].URL != "/site/programs/alpha/versions/v1-new/index.html" {
		t.Fatalf("wrong inferred URL: %s", payload.Items[0].URL)
	}
}

func TestGenerationRejectsNonLoopbackClients(t *testing.T) {
	server, _ := testServer(t)
	request := httptest.NewRequest(http.MethodPost, "/api/generate", bytes.NewReader(nil))
	request.RemoteAddr = "192.0.2.10:1234"
	response := httptest.NewRecorder()

	server.handler().ServeHTTP(response, request)

	if response.Code != http.StatusForbidden {
		t.Fatalf("got %d, want %d", response.Code, http.StatusForbidden)
	}
}

func TestGenerationValidatesPythonAndProbeJSONBeforeLaunching(t *testing.T) {
	server, _ := testServer(t)
	var body bytes.Buffer
	writer := multipart.NewWriter(&body)
	part, _ := writer.CreateFormFile("source", "program.txt")
	_, _ = part.Write([]byte("def kernel(x):\n    return x\n"))
	_ = writer.WriteField("probes", "not-json")
	_ = writer.Close()
	request := httptest.NewRequest(http.MethodPost, "/api/generate", &body)
	request.RemoteAddr = "127.0.0.1:1234"
	request.Header.Set("Content-Type", writer.FormDataContentType())
	response := httptest.NewRecorder()

	server.handler().ServeHTTP(response, request)

	if response.Code != http.StatusBadRequest {
		t.Fatalf("got %d, want %d: %s", response.Code, http.StatusBadRequest, response.Body.String())
	}
}

func TestPublishedModuleFunctionRunsThroughLoopbackPython(t *testing.T) {
	server, root := testServer(t)
	directory := filepath.Join(root, "site", "programs", "demo", "versions", "v1", "source", "python_source")
	if err := os.MkdirAll(directory, 0700); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(directory, "demo.py"), []byte("def render(): return 1\n"), 0600); err != nil {
		t.Fatal(err)
	}
	helper := "import json\nprint(json.dumps({'ok': True, 'result': {'kind': 'scalar', 'value': 1}}))\n"
	if err := os.WriteFile(filepath.Join(server.config.turing, "run_site_callable.py"), []byte(helper), 0600); err != nil {
		t.Fatal(err)
	}
	body := bytes.NewBufferString(`{"source":"/site/programs/demo/versions/v1/source/python_source/demo.py","callable":"render","arguments":{}}`)
	request := httptest.NewRequest(http.MethodPost, "/api/run", body)
	request.RemoteAddr = "127.0.0.1:1234"
	request.Header.Set("Content-Type", "application/json")
	response := httptest.NewRecorder()

	server.handler().ServeHTTP(response, request)

	if response.Code != http.StatusOK || !bytes.Contains(response.Body.Bytes(), []byte(`"ok": true`)) {
		t.Fatalf("unexpected callable response: %d %s", response.Code, response.Body.String())
	}
}

func TestStaticRootServesMainPage(t *testing.T) {
	server, _ := testServer(t)
	request := httptest.NewRequest(http.MethodGet, "/", nil)
	response := httptest.NewRecorder()

	server.handler().ServeHTTP(response, request)

	if response.Code != http.StatusOK || response.Body.String() != "homepage" {
		t.Fatalf("unexpected static response: %d %q", response.Code, response.Body.String())
	}
	if response.Header().Get("Cache-Control") != "no-cache" {
		t.Fatalf("homepage cache policy missing: %q", response.Header().Get("Cache-Control"))
	}
}
