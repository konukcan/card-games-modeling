#!/bin/bash
#
# Launch Overnight Normalization Comparison
# ==========================================
#
# This script runs the L2Norm vs LayerNorm+Scale wake-sleep comparison
# with proper system protection to prevent sleep during overnight runs.
#
# IMPORTANT: Uses pre-recorded balanced tasks with:
# - Guaranteed 50/50 positive/negative balance
# - Near-miss negatives (differ by one card)
# - Disjoint training/holdout pools
# - No spurious 0-positive tasks
#
# Expected runtime: 2-4 hours
# Default config: 35 tasks, 5 iterations, 100k enumeration budget
#
# Usage:
#   ./launch_normalization_comparison.sh           # Full experiment
#   ./launch_normalization_comparison.sh --quick   # Quick validation test
#

set -e

# Navigate to src directory
cd "$(dirname "$0")/.."

# Timestamp for log file
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="logs/normalization_comparison_${TIMESTAMP}.log"

# Create logs directory
mkdir -p logs

echo "============================================================"
echo "LAUNCHING NORMALIZATION COMPARISON EXPERIMENT"
echo "============================================================"
echo ""
echo "Timestamp: $(date)"
echo "Log file: ${LOG_FILE}"
echo ""

# Check pre-recorded tasks exist
if [ ! -f "data/prerecorded_tasks/pretraining_tasks.json" ]; then
    echo "ERROR: Pre-recorded tasks not found!"
    echo "Run: python3 generate_prerecorded_tasks.py"
    exit 1
fi

echo "Pre-recorded tasks found:"
python3 -c "
from dreamcoder_core.task_generation import load_prerecorded_tasks
from pathlib import Path
tasks = load_prerecorded_tasks(Path('data/prerecorded_tasks/pretraining_tasks.json'))
print(f'  {len(tasks)} balanced tasks loaded')
for t in tasks[:3]:
    pos = sum(1 for _, l in t.examples if l)
    neg = sum(1 for _, l in t.examples if not l)
    print(f'    {t.name}: {pos}+/{neg}-')
"

echo ""
echo "Launching with caffeinate protection..."
echo ""

# Launch with caffeinate to prevent system sleep
nohup caffeinate -d -i -s python3 experiments/compare_normalization_wakesleep.py "$@" > "${LOG_FILE}" 2>&1 &
PID=$!

echo "Process started with PID: ${PID}"
echo "Caffeinate is preventing system sleep"
echo ""
echo "To monitor progress:"
echo "  tail -f ${LOG_FILE}"
echo ""
echo "To check if still running:"
echo "  ps aux | grep ${PID}"
echo ""
echo "To view summary when done:"
echo "  cat results_normalization_wakesleep/comparison_*/summary.json | python3 -m json.tool"
echo ""

# Save PID to file for later reference
echo "${PID}" > "logs/normalization_comparison_${TIMESTAMP}.pid"

echo "PID saved to: logs/normalization_comparison_${TIMESTAMP}.pid"
echo "============================================================"
