#!/usr/bin/env sh
set -eu

state_directory="${SYMMETRY_HARNESS_HOME:-$HOME/.symmetry-harness}"
python_path_file="$state_directory/python-path.txt"

blocked() {
    printf '%s\n' '{"status":"blocked","phase":"runtime","issues":["The local symmetry-harness runtime is unavailable."],"recommendations":["Run the symmetry-harness installer once."]}'
    exit 2
}

[ -f "$python_path_file" ] || blocked
harness_python="$(tr -d '\r' < "$python_path_file")"
[ -x "$harness_python" ] || blocked

exec "$harness_python" -m symmetry_harness.cli wait --server-port 7860 "$@"
