#!/usr/bin/env sh
set -eu

state_directory="${SYMMETRY_HARNESS_HOME:-$HOME/.symmetry-harness}"
python_path_file="$state_directory/python-path.txt"
config_path_file="$state_directory/config-path.txt"

blocked() {
    printf '%s\n' '{"status":"blocked","phase":"runtime","issues":["The local symmetry-harness runtime is unavailable."],"recommendations":["Run the symmetry-harness installer once."]}'
    exit 2
}

[ -f "$python_path_file" ] || blocked
[ -f "$config_path_file" ] || blocked

IFS= read -r harness_python < "$python_path_file"
IFS= read -r config_path < "$config_path_file"

[ -x "$harness_python" ] || blocked
[ -f "$config_path" ] || blocked

exec "$harness_python" -m symmetry_harness.cli launch \
    --config "$config_path" \
    --server-port 0 \
    --no-inbrowser \
    "$@"
