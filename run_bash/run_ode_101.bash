#!/bin/bash

# Experiment: ode_101 - Running 1 experiment with nohup
# Dimension: 4D
# Experiment configuration:
#   1. sigma00_desc_de_scientist (use_gt=false)

cd "$(dirname "$0")/.." || exit

# Activate Conda environment
eval "$(conda shell.bash hook)"
conda activate ode_llm_sr

# Unset PYTHONPATH (prevent environment conflicts)
unset PYTHONPATH

# Check Python path
PYTHON_CMD=$(which python)
if [ -z "$PYTHON_CMD" ]; then
    PYTHON_CMD=$(which python3)
fi

echo "========================================="
echo "ode_101 - Running 1 experiment with nohup"
echo "Dimension: 4D"
echo "Python: $PYTHON_CMD"
echo "========================================="
echo ""

# Create log directory
LOG_DIR="run_bash/nohup_log"
mkdir -p "$LOG_DIR"
# Experiment 1: sigma00_desc_de_scientist (background execution)
echo "[START] Experiment 1/1: ode_101 - sigma00_desc_de_scientist"
nohup $PYTHON_CMD -u main.py \
    --use_var_desc true \
    --use_scientist true \
    --problem_name ode_101 \
    --use_gt false \
    --max_params 8 \
    --dim 4 \
    --evolution_num 100 \
    --de_tolerance 1e-5 \
    --bfgs_tolerance 1e-9 \
    --recursion_limit 40 \
    --timeout 240 \
    --max_retries 3 \
    --sampler_model_name "google/gemini-2.5-flash-lite" \
    --scientist_model_name "google/gemini-2.5-flash-lite" \
    > "$LOG_DIR/ode_101_experiment.log" 2>&1 &
PID1=$!
echo "  PID: $PID1"

echo ""
echo "========================================="
echo "1 experiment is running in the background."
echo "Log files:"
echo "  - $LOG_DIR/ode_101_experiment.log"
echo "========================================="
echo ""
echo "Waiting for all processes to complete..."
echo ""

# Wait for all processes to complete
wait $PID1
EXIT1=$?

echo ""
echo "========================================="
echo "Experiment results:"
echo "  1/1 experiment:  $([ $EXIT1 -eq 0 ] && echo '[SUCCESS]' || echo '[FAILED]')"
echo "========================================="

# Check if all experiments succeeded
if [ $EXIT1 -eq 0 ]; then
    echo ""
    echo "[ALL EXPERIMENTS COMPLETE] ode_101 - 1 experiment completed successfully!"
    exit 0
else
    echo ""
    echo "[WARNING] Experiment failed. Please check the log files."
    exit 1
fi
