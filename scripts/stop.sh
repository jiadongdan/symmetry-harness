#!/usr/bin/env sh
# Stop every local Symmetry Harness interface started by any tool or terminal.
# Usage:  "<skill-directory>/scripts/stop.sh"

set -u

self=$$

# Collect PIDs whose full command line contains both "symmetry_harness.cli"
# and "launch". Match on the whole command line (not a fixed argument index),
# because the invocation may be `python -m symmetry_harness.cli ...` or a
# direct console script, which shifts argument positions.
pids=$(ps -eo pid=,args= | grep 'symmetry_harness\.cli' | grep 'launch' | grep -v 'grep' | awk -v s="$self" '$1 != s {print $1}')

if [ -z "$pids" ]; then
    printf '%s\n' "No running Symmetry Harness interface found."
    exit 0
fi

for pid in $pids; do
    printf '%s\n' "Stopping PID $pid"
    kill -TERM "$pid" 2>/dev/null || kill -KILL "$pid" 2>/dev/null || true
done

sleep 2

remaining=$(ps -eo pid=,args= | grep 'symmetry_harness\.cli' | grep 'launch' | grep -v 'grep' | awk -v s="$self" '$1 != s {print $1}')
if [ -n "$remaining" ]; then
    printf '%s\n' "WARNING: process(es) still running: $remaining"
    exit 1
fi

printf '%s\n' "All Symmetry Harness interfaces stopped."
exit 0
