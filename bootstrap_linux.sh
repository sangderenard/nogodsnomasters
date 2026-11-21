#!/usr/bin/env bash
# The SpeakToMe workspace from nothing, on Linux: the same layout as C:\dev\Powershell on the Windows server, then the
# compiles. Owner, 2026-11-19: "something to set the repo filestructure up from scratch and then run compiles from an
# oracle linux instance or ubuntu with like, a script that handles everything, proven by the wsl work we do".
#
#   sudo bash bootstrap_linux.sh [--root /srv/powershell] [--user nodus] [--full] [--no-build] [--only nodus,turing,...]
#
#   1. the outer repository (nogodsnomasters) at --root; it tracks only the coordination files and a few whitelisted
#      children, so each child repository is cloned into its own folder beside them, as on the server;
#   2. the machine: nodus/tools/build_service/provision_linux.sh (packages for Ubuntu/Debian or Oracle Linux, vcpkg,
#      the service account);
#   3. the compiles, as the service account: Nodus (CMake + vcpkg, this machine's own triplet, Torch off) and Turing's
#      Python environment (turing/.venv: requirements.txt + llvmlite, setuptools, ziglang).
# Everything belongs to the service account. Clones are shallow unless --full. A second run fetches and fast-forwards
# what is already there, then builds again. The log of a cloud-init run is /var/log/speaktome-bootstrap.log.
#
# Repositories (path, remote, branch). tensor-calculator is private and socratic has no remote: both are left out
# (Nodus builds without them). wheelhouse is on hiatus and is left out.
set -euo pipefail

ROOT=/srv/powershell
USER_NAME=nodus
DEPTH=(--depth 1)
BUILD=1
ONLY=""
while [ $# -gt 0 ]; do
  case "$1" in
    --root) ROOT="$2"; shift 2 ;;
    --user) USER_NAME="$2"; shift 2 ;;
    --full) DEPTH=(); shift ;;
    --no-build) BUILD=0; shift ;;
    --only) ONLY=",$2,"; shift 2 ;;
    *) echo "unknown argument $1" >&2; exit 2 ;;
  esac
done
[ "$(id -u)" -eq 0 ] || { echo "run as root (sudo)" >&2; exit 2; }

REPOS=(
  "speaktome          https://github.com/sangderenard/speaktome.git          nogodsnomasters"
  "turing             https://github.com/sangderenard/Turing.git             main"
  "nodus              https://github.com/sangderenard/nodus.git              nogodsnomasters"
  "spectral-analyzer  https://github.com/sangderenard/spectral-analyzer.git  nogodsnomasters"
  "geometry           https://github.com/sangderenard/geometry.git           nogodsnomasters"
  "amp                https://github.com/sangderenard/amp.git                main"
  "fftfree            https://github.com/sangderenard/fftfree.git            nogodsnomasters"
  "meta-nn            https://github.com/sangderenard/meta-nn.git            nogodsnomasters"
  "pigeon             https://github.com/sangderenard/pigeon.git             main"
  "lfsavoider         https://github.com/sangderenard/lfsavoider.git         nogodsnomasters"
  "pcb                https://github.com/sangderenard/electrofluid.git       main"
)
OUTER="https://github.com/sangderenard/nogodsnomasters.git nogodsnomasters"

# A minimal base for git itself; the full toolchain comes with the machine step.
if command -v apt-get >/dev/null 2>&1; then
  DEBIAN_FRONTEND=noninteractive apt-get install -y -q git ca-certificates curl >/dev/null
else
  dnf install -y -q git ca-certificates curl >/dev/null
fi
id "$USER_NAME" >/dev/null 2>&1 || useradd --system --create-home --home-dir "/home/$USER_NAME" --shell /bin/bash "$USER_NAME"
as_user() { sudo -u "$USER_NAME" -H "$@"; }

# Clone or fast-forward one repository into a folder.
sync_repo() {
  local dir="$1" url="$2" branch="$3"
  if [ -d "$dir/.git" ]; then
    as_user git -C "$dir" fetch --quiet "${DEPTH[@]}" origin "$branch"
    as_user git -C "$dir" checkout --quiet "$branch"
    as_user git -C "$dir" merge --ff-only --quiet "origin/$branch"
  else
    as_user git clone --quiet "${DEPTH[@]}" --branch "$branch" "$url" "$dir"
  fi
  echo "  $(basename "$dir"): $(as_user git -C "$dir" rev-parse --short HEAD) ($branch)"
}

echo "== workspace at $ROOT"
mkdir -p "$ROOT"
chown "$USER_NAME:$USER_NAME" "$ROOT"
if [ ! -d "$ROOT/.git" ]; then
  # The outer repository goes into the (empty) root itself; its .gitignore keeps the children opaque.
  set -- $OUTER
  as_user git clone --quiet "${DEPTH[@]}" --branch "$2" "$1" "$ROOT/.outer-clone"
  as_user bash -c "shopt -s dotglob; mv '$ROOT/.outer-clone'/* '$ROOT'/ && rmdir '$ROOT/.outer-clone'"
  echo "  outer: $(as_user git -C "$ROOT" rev-parse --short HEAD) ($2)"
else
  set -- $OUTER
  sync_repo "$ROOT" "$1" "$2"
fi
for line in "${REPOS[@]}"; do
  set -- $line
  [ -z "$ONLY" ] || [[ "$ONLY" == *",$1,"* ]] || continue
  sync_repo "$ROOT/$1" "$2" "$3"
done
as_user mkdir -p "$ROOT/srv/state" "$ROOT/srv/state/secrets" "$ROOT/srv/state/content" "$ROOT/srv/state/logs"
chmod 700 "$ROOT/srv/state/secrets"

echo "== machine"
bash "$ROOT/nodus/tools/build_service/provision_linux.sh" --user "$USER_NAME"
TRIPLET="$(cat /etc/nodus/triplet)"

[ "$BUILD" -eq 1 ] || { echo "== done (no build)"; exit 0; }

echo "== Nodus ($TRIPLET)"
mkdir -p /srv/nodus-vcpkg-cache && chown "$USER_NAME:$USER_NAME" /srv/nodus-vcpkg-cache
FORCE=""; case "$TRIPLET" in arm64-*) FORCE=1 ;; esac
as_user env VCPKG_DEFAULT_BINARY_CACHE=/srv/nodus-vcpkg-cache VCPKG_FORCE_SYSTEM_BINARIES="$FORCE" \
  cmake -S "$ROOT/nodus" -B "$ROOT/nodus/build-linux" -G Ninja -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_TOOLCHAIN_FILE=/opt/vcpkg/scripts/buildsystems/vcpkg.cmake -DVCPKG_TARGET_TRIPLET="$TRIPLET" \
    -DNODUS_ENABLE_TORCH=OFF
as_user env VCPKG_FORCE_SYSTEM_BINARIES="$FORCE" cmake --build "$ROOT/nodus/build-linux" --target nodus_headless

echo "== Turing's Python environment"
as_user python3 -m venv "$ROOT/turing/.venv"
as_user "$ROOT/turing/.venv/bin/pip" install --quiet --upgrade pip
# Beyond requirements.txt, the compile path needs three packages (first Oracle run of the dt-system compile,
# 2026-11-21: each was a wall before any compiling):
#   llvmlite   -- the LLVM lane's IR builder;
#   setuptools -- cffi builds the C backend through it, and a Python 3.12+ venv no longer ships it or distutils;
#   ziglang    -- the bundled clang the LLVM and C lanes link with (`python -m ziglang cc`), on every platform.
as_user "$ROOT/turing/.venv/bin/pip" install --quiet -r "$ROOT/turing/requirements.txt" llvmlite setuptools ziglang
as_user "$ROOT/turing/.venv/bin/python" -c "import llvmlite, setuptools, sympy, numpy, ziglang; print('  turing env: llvmlite', llvmlite.__version__, 'setuptools', setuptools.__version__)"
as_user "$ROOT/turing/.venv/bin/python" -m ziglang version

echo "== done: $ROOT (Nodus in nodus/build-linux, Turing in turing/.venv)"
