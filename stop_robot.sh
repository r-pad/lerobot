#!/usr/bin/env bash
# Stop leftover robot data-collection processes (e.g. a run suspended with Ctrl+Z) and free the
# deoxys ZMQ ports and cameras.
#
# Usage: bash stop_robot.sh          # stop everything, then report
#        bash stop_robot.sh --dry    # only show what would be stopped
#
# Signals: SIGTERM, then SIGCONT so suspended (state T) processes wake up and act on it; SIGKILL
# only for whatever is still alive after a few seconds. Python exits on SIGTERM without running
# the control loop again, so no new commands reach the robot.

PORTS="5555 5556 5557 5558 5559"
PATTERN="collect_data.sh|lerobot/scripts/control_robot.py"

# All descendants of a pid (image-writer workers, pixi -> python, ...).
descendants() {
    local child
    for child in $(pgrep -P "$1"); do
        echo "$child"
        descendants "$child"
    done
}

pids=""
for pid in $(pgrep -u "$USER" -f "$PATTERN"); do
    pids="$pids $pid $(descendants "$pid")"
done
for port in $PORTS; do
    pids="$pids $(ss -ltnpH "sport = :$port" 2>/dev/null | grep -o 'pid=[0-9]*' | cut -d= -f2)"
done
pids=$(echo "$pids" | tr ' ' '\n' | grep -E '^[0-9]+$' | grep -vx "$$" | sort -un | xargs)

if [ -z "$pids" ]; then
    echo "No leftover robot processes found."
else
    echo "Robot processes:"
    ps -o pid=,stat=,etime=,args= -p "${pids// /,}" | cut -c1-150
    if [ "$1" = "--dry" ]; then
        echo "(dry run: nothing stopped)"
        exit 0
    fi
    kill -TERM $pids 2>/dev/null
    kill -CONT $pids 2>/dev/null
    for _ in $(seq 1 10); do
        alive=$(ps -o pid= -p "${pids// /,}" 2>/dev/null | xargs)
        [ -z "$alive" ] && break
        sleep 1
    done
    if [ -n "$alive" ]; then
        echo "Still alive after SIGTERM, sending SIGKILL: $alive"
        kill -KILL $alive 2>/dev/null
        sleep 1
    fi
    echo "Stopped."
fi

busy=$(for port in $PORTS; do ss -ltnH "sport = :$port" 2>/dev/null; done)
if [ -n "$busy" ]; then
    echo "WARNING: ports still in use:"
    echo "$busy"
else
    echo "Ports $PORTS are free."
fi
cams=$(fuser /dev/video* 2>/dev/null | xargs)
if [ -n "$cams" ]; then
    echo "WARNING: cameras still held by pids: $cams"
else
    echo "Cameras are free."
fi
