#!/bin/bash

# Experiment: ode_045, ode_047, ode_051 - Parallel execution
# Dimension: 2D
# Experiment configuration:
#   1. ode_045
#   2. ode_047
#   3. ode_051

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
echo "ode_045, ode_047, ode_051 - Parallel experiments with nohup"
echo "Dimension: 2D"
echo "Python: $PYTHON_CMD"
echo "========================================="
echo ""

# Create log directory
LOG_DIR="run_bash/nohup_log"
mkdir -p "$LOG_DIR"

# Common configuration variables
COMMON_ARGS="--use_var_desc true --use_scientist true --use_gt false --max_params 8 --dim 2 --evolution_num 100 --de_tolerance 1e-5 --bfgs_tolerance 1e-9 --recursion_limit 40 --timeout 240 --max_retries 3 --sampler_model_name google/gemini-2.5-flash-lite --scientist_model_name google/gemini-2.5-flash-lite"

# Experiment 1: ode_045
echo "[START] Experiment 1/3: ode_045"
nohup $PYTHON_CMD -u main.py $COMMON_ARGS --problem_name ode_045 > "$LOG_DIR/ode_045_experiment.log" 2>&1 &
PID1=$!
echo "  PID: $PID1"

# Experiment 2: ode_047
echo "[START] Experiment 2/3: ode_047"
nohup $PYTHON_CMD -u main.py $COMMON_ARGS --problem_name ode_047 > "$LOG_DIR/ode_047_experiment.log" 2>&1 &
PID2=$!
echo "  PID: $PID2"

# Experiment 3: ode_051
echo "[START] Experiment 3/3: ode_051"
nohup $PYTHON_CMD -u main.py $COMMON_ARGS --problem_name ode_051 > "$LOG_DIR/ode_051_experiment.log" 2>&1 &
PID3=$!
echo "  PID: $PID3"

echo ""
echo "========================================="
echo "3 experiments are running in parallel in the background."
echo "Log files:"
echo "  - $LOG_DIR/ode_045_experiment.log"
echo "  - $LOG_DIR/ode_047_experiment.log"
echo "  - $LOG_DIR/ode_051_experiment.log"
echo "========================================="
echo ""
echo "Waiting for all processes to complete..."
echo ""

# Wait for all processes to complete
wait $PID1
EXIT1=$?

wait $PID2
EXIT2=$?

wait $PID3
EXIT3=$?

echo ""
echo "========================================="
echo "Experiment results:"
echo "  ode_045: $([ $EXIT1 -eq 0 ] && echo '[SUCCESS]' || echo '[FAILED]')"
echo "  ode_047: $([ $EXIT2 -eq 0 ] && echo '[SUCCESS]' || echo '[FAILED]')"
echo "  ode_051: $([ $EXIT3 -eq 0 ] && echo '[SUCCESS]' || echo '[FAILED]')"
echo "========================================="

# Check overall success
if [ $EXIT1 -eq 0 ] && [ $EXIT2 -eq 0 ] && [ $EXIT3 -eq 0 ]; then
    echo ""
    echo "[ALL EXPERIMENTS COMPLETE] All experiments completed successfully!"
    exit 0
else
    echo ""
    echo "[WARNING] Some experiments failed. Please check the log files."
    exit 1
fi
