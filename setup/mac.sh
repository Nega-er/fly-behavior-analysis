#!/bin/zsh -f
set -eu
ROOT="${0:A:h:h}"
# Start a clean child process. Do not deactivate or modify a user's Conda shell.
if [[ "${FLY_CLEAN_RUNTIME:-}" != 1 ]]; then
  conda_detected=0
  if [[ -n "${CONDA_PREFIX:-}${CONDA_DEFAULT_ENV:-}" ]] || [[ "${CONDA_SHLVL:-0}" == <-> && "${CONDA_SHLVL:-0}" -gt 0 ]]; then conda_detected=1; fi
  exec /usr/bin/env -i HOME="$HOME" USER="${USER:-}" LOGNAME="${LOGNAME:-}" \
    PATH=/usr/bin:/bin:/usr/sbin:/sbin TMPDIR="${TMPDIR:-/tmp}" LANG=en_US.UTF-8 \
    FLY_CLEAN_RUNTIME=1 FLY_CONDA_DETECTED="$conda_detected" /bin/zsh -f "$0" "$@"
fi
mkdir -p "$ROOT/.runtime/logs" "$ROOT/.runtime/tools" "$ROOT/.runtime/cache"
export UV_PYTHON_INSTALL_DIR="$ROOT/.runtime/python"
export UV_CACHE_DIR="$ROOT/.runtime/cache"
export UV_PYTHON_BIN_DIR="$ROOT/.runtime/tools"
export UV_NO_MODIFY_PATH=1
export UV_PYTHON_DOWNLOADS=automatic
export UV_SYSTEM_CERTS=true
export PYTHONNOUSERSITE=1
export PYTHONUTF8=1
lock="$ROOT/setup/runtime-lock.json"
uv_version=$(/usr/bin/plutil -extract uv_version raw -o - "$lock")
python_version=$(/usr/bin/plutil -extract python_version raw -o - "$lock")
case "$(/usr/bin/uname -m)" in
  arm64) target=mac_arm64 ;;
  x86_64) target=mac_x86_64 ;;
  *) print 'Unsupported Mac architecture.' >&2; exit 1 ;;
esac
archive=$(/usr/bin/plutil -extract "platforms.$target.archive" raw -o - "$lock")
directory=$(/usr/bin/plutil -extract "platforms.$target.directory" raw -o - "$lock")
expected=$(/usr/bin/plutil -extract "platforms.$target.sha256" raw -o - "$lock")
uv="$ROOT/.runtime/tools/uv-$uv_version"
if [[ ! -x "$uv" ]] || [[ "$("$uv" --version | /usr/bin/awk '{print $2}')" != "$uv_version" ]]; then
  print "Downloading verified uv $uv_version…"
  temporary=$(/usr/bin/mktemp -d "$ROOT/.runtime/cache/bootstrap.XXXXXX")
  trap '/bin/rm -rf "$temporary"' EXIT
  /usr/bin/curl --proto '=https' --tlsv1.2 --fail --location --retry 3 \
    "https://github.com/astral-sh/uv/releases/download/$uv_version/$archive" -o "$temporary/$archive"
  actual=$(/usr/bin/shasum -a 256 "$temporary/$archive" | /usr/bin/awk '{print $1}')
  [[ "$actual" == "$expected" ]] || { print 'uv checksum mismatch.' >&2; exit 1; }
  /usr/bin/tar -xzf "$temporary/$archive" -C "$temporary"
  /bin/cp "$temporary/$directory/uv" "$uv"
  /bin/chmod 755 "$uv"
fi
print "Conda inherited: ${FLY_CONDA_DETECTED:-0}; child runtime: isolated"
"$uv" python install "$python_version" --no-bin
python_path=$("$uv" python find --managed-python "$python_version")
exec "$python_path" "$ROOT/setup/runtime_manager.py" --uv "$uv" "$@"
