#!/bin/bash
# Overnight Pretraining Launcher (Option C+ Enhanced)
# Uses caffeinate to prevent system sleep and nohup for persistence
# Estimated runtime: 10-12 hours

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

TIMESTAMP=$(date +"%Y-%m-%dT%H-%M-%S")
LOG_DIR="results/overnight_pretraining"
LOG_FILE="${LOG_DIR}/console_log_${TIMESTAMP}.log"
PID_FILE="${LOG_DIR}/overnight.pid"

echo "========================================"
echo "OVERNIGHT PRETRAINING LAUNCHER"
echo "(Option C+ Enhanced - Staged Curriculum)"
echo "========================================"
echo "Started at: $(date)"
echo "Log file: $LOG_FILE"
echo ""
echo "Training Configuration:"
echo "  Phase 1 (iters 1-5):   22 easy rules, budget=200K, depth=8"
echo "  Phase 2 (iters 6-12):  43 all rules, budget=300K, depth=9"
echo "  Phase 3 (iters 13-20): 43 all rules, budget=500K, depth=10"
echo ""
echo "Estimated runtime: 10-12 hours"
echo ""

# Create results directory
mkdir -p "$LOG_DIR"

# Kill any existing overnight experiments
if [ -f "$PID_FILE" ]; then
    OLD_PID=$(cat "$PID_FILE")
    if ps -p "$OLD_PID" > /dev/null 2>&1; then
        echo "Killing existing experiment (PID: $OLD_PID)..."
        kill "$OLD_PID" 2>/dev/null
        sleep 2
    fi
    rm -f "$PID_FILE"
fi

# Start the experiment with caffeinate (prevents sleep) and nohup (survives terminal close)
echo "Starting pretraining with caffeinate (prevents system sleep)..."
echo ""

# Use caffeinate to prevent:
# -i: idle sleep
# -m: disk sleep
# -d: display sleep
# -s: system sleep (requires AC power for laptops)
# Run python with unbuffered output (-u) in background, redirect output to log file
nohup caffeinate -dims python3 -u run_overnight_pretraining.py > "$LOG_FILE" 2>&1 &
EXPERIMENT_PID=$!

echo "$EXPERIMENT_PID" > "$PID_FILE"

echo "Pretraining started!"
echo "  PID: $EXPERIMENT_PID"
echo "  Log: $LOG_FILE"
echo ""
echo "To monitor progress:"
echo "  tail -f $LOG_FILE"
echo ""
echo "To check if running:"
echo "  ps -p $EXPERIMENT_PID"
echo ""
echo "To stop:"
echo "  kill $EXPERIMENT_PID"
echo ""
echo "Results will be saved to:"
echo "  $LOG_DIR/"
echo ""
echo "========================================"
echo "You can now close this terminal safely."
echo "The system will stay awake until complete."
echo "========================================"
