#!/bin/bash
# Start the overnight pre-training with PyPy-accelerated workers
# This script uses caffeinate to prevent the Mac from sleeping

cd "$(dirname "$0")"

LOG_FILE="results/overnight_cython/run_$(date +%Y%m%d_%H%M%S).log"
mkdir -p results/overnight_cython

echo "Starting overnight pre-training with PyPy workers..."
echo "Log file: $LOG_FILE"
echo ""
echo "Optimizations enabled:"
echo "  - Early pruning: YES"
echo "  - PyPy workers: YES (4 parallel)"
echo ""
echo "Expected runtime: 4-6 hours"
echo ""

# Run with caffeinate to prevent sleep, redirect output to log file
# Using nohup so it continues even if terminal closes
nohup caffeinate -dims python3 -u run_overnight_cython.py > "$LOG_FILE" 2>&1 &

PID=$!
echo "Process started with PID: $PID"
echo ""
echo "To monitor progress:"
echo "  tail -f $LOG_FILE"
echo ""
echo "To check if still running:"
echo "  ps aux | grep run_overnight"
