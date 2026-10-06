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
harness_python="$(tr -d '\r' < "$python_path_file")"
config_path="$(tr -d '\r' < "$config_path_file")"
[ -x "$harness_python" ] || blocked
[ -f "$config_path" ] || blocked

log_directory="$state_directory/logs"
mkdir -p "$log_directory"
stdout_path="$log_directory/fast-open.stdout.json"
stderr_path="$log_directory/fast-open.stderr.log"
: > "$stdout_path"
: > "$stderr_path"

nohup "$harness_python" -m symmetry_harness.cli launch \
    --config "$config_path" \
    --server-port 7860 \
    --no-inbrowser \
    "$@" >"$stdout_path" 2>"$stderr_path" </dev/null &
launch_pid=$!

exec "$harness_python" -m symmetry_harness.cli wait \
    --server-port 7860 \
    --timeout-seconds 180 \
    --launch-pid "$launch_pid" \
    --launch-output "$stdout_path" \
    --launch-error "$stderr_path"
