# PUBLISHING BUNDLES TO ROOT

How to compile a Turing program into a browser-runnable bundle and get it
onto the live, publicly-checkable page — as opposed to building it and
leaving it stranded inside `turing/`'s own local output, where root-relative
paths don't resolve and nobody but the machine that built it can see it.

## The two roots, and why this matters

There are two separate Git repositories in play:

- **`C:\dev\Powershell\turing\`** — the compiler. Has its own `.git`, its own
  history, its own local `turing/site/` scratch output from tests and ad hoc
  runs.
- **`C:\dev\Powershell\`** (this directory) — the coordination repository,
  remote [`sangderenard/nogodsnomasters`](https://github.com/sangderenard/nogodsnomasters),
  branch `nogodsnomasters`. Its `.gitignore` treats every child directory as
  opaque **except one**:

  ```gitignore
  /*/
  !/site/
  ```

  `/site/` (i.e. `C:\dev\Powershell\site\`) is "the versioned GitHub Pages
  runtime" — the one deliberate child directory this repo tracks. The root
  `index.html` plus everything under `site/` is what actually gets served.

A bundle built into `turing/site/...` is real, but it's invisible to
anyone checking the live page: nothing under `turing/` is served, and
root-relative paths (`/site/programs/...`) only resolve correctly against
this repo's root, not a local `file://` path or a mismatched directory
structure. **"Publish" means the compiled bundle has to land under
`C:\dev\Powershell\site\`, not under `C:\dev\Powershell\turing\site\`.**

The compiler already knows this distinction. In
`turing/src/compiler/site_bundle.py`:

```python
TURING_REPOSITORY_ROOT = Path(__file__).resolve().parents[2]   # .../turing
DEFAULT_PUBLISH_ROOT = TURING_REPOSITORY_ROOT.parent            # C:\dev\Powershell
```

`resolve_publish_root()` actively **rejects** `turing/` itself as a
destination:

```python
def resolve_publish_root(destination):
    resolved = Path(destination).resolve()
    if resolved == TURING_REPOSITORY_ROOT:
        raise ValueError(
            "gallery bundles belong in the parent workspace root, not Turing"
        )
    return resolved
```

Every CLI builder below defaults `--destination` to `DEFAULT_PUBLISH_ROOT`
already, so **running them from inside `turing/` with no flags publishes to
the right place by default.** You only need `--destination` to point
somewhere else on purpose.

## Building and publishing a bundle

Run from inside `turing/` (a Python venv with the compiler's dependencies
active).

### One arbitrary Python source → one bundle

```bash
python build_site_page.py --source path/to/your_program.py
```

Common flags (`turing/build_site_page.py`):

| Flag | Meaning | Default |
|---|---|---|
| `--source` | path to the `.py` file to compile (required) | — |
| `--destination` | publish root | `DEFAULT_PUBLISH_ROOT` (`C:\dev\Powershell`) |
| `--entrypoint` | which function to compile | inferred (or from a `TURING_PAGE` dict in the source) |
| `--title`, `--slug` | page identity | inferred |
| `--probes-json` | feed values as a JSON object, for a source with no literal `TURING_PAGE` probes | `{}` |
| `--no-backends` | skip the multi-backend source tabs (C/GLSL/Fortran/WebGL/WebGPU/...) | included by default |
| `--no-mathematics` | skip the SymPy math-model panel | included by default |
| `--result-json` | also write the JSON result (bundle/page/url paths) to a file | stdout only |

It prints progress lines as it compiles, then a JSON summary on the final
line: `{"ok": true, "bundle": "...", "page": "...", "url": "/site/programs/<slug>/versions/<version>/", "manifest": {...}}`.

Internally this calls `build_program_bundle(source, destination, ...)` from
`site_bundle.py`, which writes the standard layout:

```text
site/programs/<slug>/versions/<version>/
  bundle.json
  index.html
  source/python_source/<file>.py
  source/<backend>/<artifact>       # webgl/, webgpu/, glsl/, fortran/, wat/, ...
  wasm/<module>.wasm
  math/sympy-process-model.json
  build/compiler.log
```

Content-addressed by source + compile settings: rebuilding the same source
with the same configuration reuses the existing version directory instead
of creating a duplicate.

### The fixed homepage demo

```bash
python build_homepage.py
# or explicitly:
python build_homepage.py --destination C:/dev/Powershell
```

Compiles the Mandelbrot homepage kernel through every backend and writes
the homepage shell, same default-to-root behavior as above.

### Other publishers

- `turing/build_wasm_compiler_page.py` — publishes the WASM-compiler
  inspection page itself (fixed slug `turing-webassembly-compiler`), same
  `--destination` default.
- `python -c "..."` / a short script calling `site_bundle.build_program_bundle(...)`
  directly — the CLI scripts are thin wrappers over this function; anything
  they can do, calling it directly can do too, with more control (e.g.
  `presentation_shader`, `bake_mode`, `progress_sink`).

## Checking the result

**Locally**, before pushing anything: from `C:\dev\Powershell` (the root, in
a Go environment):

```powershell
go run .
```

Open `http://localhost:8787`. This is a loopback-only server that walks
`site/` for `bundle.json` files and serves whatever's already there — no
GitHub round-trip needed, but it only reflects what's on disk locally, and
(per prior session notes) some root-relative page behavior only works
correctly when actually served from a root — this local server exists
precisely to be that root, so prefer it over opening a bundle's `index.html`
directly as a `file://` URL.

**Online**: commit and push the new/changed files under `C:\dev\Powershell\site\`
(and root `index.html` if the homepage changed) to `origin` on the
`nogodsnomasters` branch. This repository is a GitHub Pages source — check
this repo's GitHub *Settings → Pages* for the exact serving URL/branch
configuration if unsure; there is no GitHub Actions deploy workflow in this
repo, so Pages is serving the pushed branch content directly, not built by CI.

```bash
git add site/ index.html   # only what actually changed
git status                  # review before committing anything
git commit -m "Publish <what changed>"
git push origin nogodsnomasters
```

Follow the standard git safety practice: review `git status`/`git diff`
before staging, never blanket `git add -A` (the whole point of this repo's
`.gitignore` is to keep every other child directory out), and don't push
without confirming that's actually intended — this updates a live, public
page.

## Pitfalls

- **Passing `--destination` pointed at `turing/`** — rejected with
  `ValueError: gallery bundles belong in the parent workspace root, not Turing`.
  This is deliberate, not a bug to work around.
- **Building without publishing** — running compiler code paths that write
  into `turing/site/`, `turing/build/`, or a tmp directory (as tests do)
  produces real output that will never be reachable online. If the goal is
  "check this online," it has to go through `build_program_bundle`/
  `build_site_page.py` with a destination that resolves to
  `C:\dev\Powershell`, and then actually get committed and pushed.
- **Forgetting to push** — a bundle written to `C:\dev\Powershell\site\` is
  not live until it's committed and pushed to `origin nogodsnomasters`; local
  disk state and the public page are two different things.
