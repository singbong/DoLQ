# DoLQ: Discovering Ordinary Differential Equations with LLM-Based Qualitative and Quantitative Evaluation

This repository contains the official implementation of **DoLQ**, a multi-agent framework for discovering governing ordinary differential equations (ODEs) from observational data. DoLQ integrates large language model (LLM)-based qualitative reasoning with quantitative evaluation to identify interpretable and physically plausible differential equations.

---

## Abstract

Discovering governing differential equations from observational data is a fundamental challenge in scientific machine learning. Existing symbolic regression approaches rely primarily on quantitative metrics; however, real-world differential equation modeling also requires incorporating domain knowledge to ensure physical plausibility. DoLQ addresses this gap by employing a multi-agent architecture: a **Sampler Agent** proposes dynamic system candidates, a **Parameter Optimizer** refines equations for accuracy, and a **Scientist Agent** leverages an LLM to conduct both qualitative and quantitative evaluations and synthesize their results to iteratively guide the search.

---

## Framework Overview

DoLQ operates through an iterative loop among three components:

```
                     +------------------+
                     |   Sampler Agent  |
                     |  (LLM-based)     |
                     +--------+---------+
                              |
                              | Candidate Terms
                              v
                     +------------------+
                     |    Parameter     |
                     |    Optimizer     |
                     |  (DE + BFGS)     |
                     +--------+---------+
                              |
                              | Optimized Equations
                              v
                     +------------------+
                     |  Scientist Agent |
                     |  (LLM-based)     |
                     +--------+---------+
                              |
                              | Feedback (Keep/Hold/Remove)
                              v
                     (Next Iteration)
```

| Component | Description |
|-----------|-------------|
| **Sampler Agent** | Proposes candidate terms with physical justifications based on system description and Scientist feedback. |
| **Parameter Optimizer** | Constructs executable functions and optimizes coefficients via hybrid DE+BFGS strategy. |
| **Scientist Agent** | Evaluates each term through qualitative semantic assessment and quantitative ablation analysis. |

---

## Methodology

### Sampler Agent

The Sampler Agent is an LLM-based component that generates candidate ODE terms for each dimension, along with natural language justifications grounded in the system description.

**Input Prompt Components:**
1. System description providing domain-specific context
2. Scientist Agent guidance (accumulated knowledge, term evaluations, removed term skeletons)
3. Technical constraints and output format requirements

**Output:** Structured JSON containing term lists with physical reasoning for each dimension.

*Implementation:* `prompt.py` (prompt construction), `evolution.py` (sampler node)

---

### Parameter Optimizer

The Parameter Optimizer transforms proposed terms into executable differential equations and finds optimal coefficients.

**Hybrid Optimization Strategy:**
1. **Differential Evolution (DE):** Global search to identify promising parameter regions
2. **BFGS:** Local refinement for precise convergence

The framework evaluates all three strategies (DE only, BFGS only, DE+BFGS) and selects the candidate with the lowest residual MSE.

*Implementation:* `optimization.py`

---

### Scientist Agent

The Scientist Agent performs comprehensive evaluation from two perspectives:

**Quantitative Evaluation (Ablation Study):**
- Temporarily removes each term by setting its coefficient to zero
- Classifies terms as `good` (positive contribution), `neutral` (negligible impact), or `bad` (negative impact)

**Qualitative Evaluation (Semantic Assessment):**
- Evaluates alignment with physical principles described in the system description
- Assigns semantic quality grades: `good`, `neutral`, or `bad`

**Feedback Synthesis:**
- `keep`: semantic quality is `good` AND performance impact is `good`
- `remove`: semantic quality is `bad` OR two consecutive `hold` actions
- `hold`: all other combinations

Removed term skeletons are recorded to prevent re-proposal in subsequent iterations.

*Implementation:* `evolution.py` (scientist analysis node), `prompt.py` (scientist prompt)

---

## Installation

### Environment Setup

```bash
# Create conda environment
make setup

# Or manually:
bash install_env.sh
```

### Dependencies

```bash
conda activate ode_llm_sr
pip install -r requirements.txt
```

### API Configuration

Create a `.env` file with your OpenRouter API key:

```
OPENROUTER_API_KEY=your_api_key_here
```

---

## Usage

### Single Experiment

```bash
python main.py \
    --problem_name ode_033 \
    --dim 2 \
    --max_params 8 \
    --evolution_num 100 \
    --use_scientist true \
    --use_differential_evolution true \
    --use_var_desc true
```

### Batch Execution

```bash
cd run_bash
nohup ./run_experiment.bash > experiment.log 2>&1 &
```

---

## Command-Line Arguments

| Argument | Required | Default | Description |
|----------|----------|---------|-------------|
| `--problem_name` | Yes | - | Problem identifier (e.g., `ode_033`) |
| `--dim` | Yes | - | System dimension (1, 2, 3, or 4) |
| `--max_params` | Yes | - | Maximum number of parameters per equation |
| `--evolution_num` | Yes | - | Number of evolution iterations |
| `--use_scientist` | No | `false` | Enable Scientist Agent for qualitative evaluation |
| `--use_differential_evolution` | No | `true` | Enable hybrid DE+BFGS optimization |
| `--use_var_desc` | No | `false` | Include system description in prompts |
| `--sampler_model_name` | No | `google/gemini-2.5-flash-lite` | LLM for Sampler Agent |
| `--scientist_model_name` | No | `google/gemini-2.5-flash-lite` | LLM for Scientist Agent |
| `--num_equations` | No | `3` | Number of candidate equations per iteration |
| `--de_tolerance` | No | `1e-5` | Differential Evolution convergence tolerance |
| `--bfgs_tolerance` | No | `1e-9` | BFGS convergence tolerance |
| `--forget_prob` | No | `0.01` | Probability to re-explore removed terms |

---

## Configuration

Key hyperparameters are defined in `config.py`:

| Parameter | Value | Description |
|-----------|-------|-------------|
| `DE_MAXITER` | 1000 | Maximum DE iterations |
| `DE_POPSIZE` | 20 | DE population size |
| `SAMPLER_TEMPERATURE` | 0.9 | Sampling temperature for Sampler LLM |
| `SCIENTIST_TEMPERATURE` | 0.6 | Sampling temperature for Scientist LLM |
| `REMOVED_TERMS_FORGET_PROBABILITY` | 0.01 | Soft forgetting probability |

---

## Directory Structure

```
ode_llm_sr/
├── main.py                  # Entry point
├── evolution.py             # LangGraph-based evolution framework
├── optimization.py          # DE and BFGS optimization
├── prompt.py                # Prompt templates for Sampler and Scientist
├── config.py                # Hyperparameters and constants
├── utils.py                 # Utility functions
├── io_utils.py              # I/O and logging utilities
├── data_loader.py           # Dataset loading
├── buffer.py                # Experience buffer (optional)
├── compare.py               # Candidate comparison logic
├── with_structured_output.py # Pydantic schemas for LLM outputs
├── Makefile                 # Environment setup automation
├── install_env.sh           # Environment installation script
├── requirements.txt         # Python dependencies
├── data/                    # Benchmark datasets
│   ├── 1D/
│   ├── 2D/
│   └── 4D/
├── logs/                    # Experiment outputs
│   └── {problem}/{model}/{timestamp}/
│       ├── iteration_json/
│       └── report/
└── run_bash/                # Batch execution scripts
```

---

## Output

Upon completion, experiments generate the following outputs in the `logs/` directory:

- `final_report.txt`: Summary including configuration, best equations, and Scientist analysis
- `final_report.json`: Machine-readable experiment summary
- `generated_equations.json`: All candidate equations and scores per iteration

---

## License

MIT License
