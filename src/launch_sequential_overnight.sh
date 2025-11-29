#!/bin/bash
#
# Sequential Overnight Run Launcher
#

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
LOG_FILE="${SCRIPT_DIR}/results/sequential_launcher.log"
CURRENT_RUN_PID=65019

mkdir -p "${SCRIPT_DIR}/results"
mkdir -p "${SCRIPT_DIR}/results/overnight_smallhands"

log_msg() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "$LOG_FILE"
}

log_msg "=========================================="
log_msg "SEQUENTIAL OVERNIGHT LAUNCHER STARTED"
log_msg "=========================================="
log_msg "Current run PID: $CURRENT_RUN_PID"
log_msg "Script directory: $SCRIPT_DIR"

# Check if current run is still active
if ! ps -p $CURRENT_RUN_PID > /dev/null 2>&1; then
    log_msg "WARNING: Current run (PID $CURRENT_RUN_PID) not found!"
    RUNNING_PID=$(pgrep -f "run_overnight_cython.py" 2>/dev/null || true)
    if [ -n "$RUNNING_PID" ]; then
        log_msg "Found running process: $RUNNING_PID"
        CURRENT_RUN_PID=$RUNNING_PID
    else
        log_msg "No current run found. Starting smaller-hands immediately."
        CURRENT_RUN_PID=""
    fi
fi

# Monitor current run
if [ -n "$CURRENT_RUN_PID" ]; then
    log_msg "Monitoring current run (PID $CURRENT_RUN_PID)..."

    while ps -p $CURRENT_RUN_PID > /dev/null 2>&1; do
        sleep 60
    done

    log_msg "Current run completed!"
    sleep 30
fi

log_msg "=========================================="
log_msg "STARTING SMALLER-HANDS CURRICULUM RUN"
log_msg "=========================================="

cd "$SCRIPT_DIR"

caffeinate -i -s -d python3 -u run_overnight_smallhands.py \
    > "${SCRIPT_DIR}/results/overnight_smallhands/run_$(date '+%Y%m%d_%H%M%S').log" 2>&1 &

SMALLHANDS_PID=$!
log_msg "Smaller-hands run started with PID: $SMALLHANDS_PID"

wait $SMALLHANDS_PID
EXIT_CODE=$?

log_msg "=========================================="
log_msg "SMALLER-HANDS RUN COMPLETED (exit: $EXIT_CODE)"
log_msg "=========================================="
