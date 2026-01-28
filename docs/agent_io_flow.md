# LLM Agent Full Prompts

This document shows the complete prompts sent to and received from LLM agents, based on actual logs from the glider descent experiment (ode_033).

---

## 1. Sampler Agent - Input Prompt
This is the actual prompt that includes Scientist feedback and removal lists.

**Condition**: Glider Descent Experiment (ode_033), Iteration 2/10

```text
You are a mathematician specializing in differential equation (ODE) discovery.

# System Description
This experiment is conducted to determine the trajectory and dynamic characteristics 
of a glider with a fixed wing design and angle of attack as it descends under the 
influence of gravity within an atmosphere containing air resistance... 
(중략) ... dimensionless velocity x0 and the flight path angle x1.

### SCIENTIST AGENT GUIDANCE
The Scientist agent has analyzed previous experiments and provides the following guidance:

#### Accumulated Knowledge (Theory)
- Analysis up to this point suggests that the -1.2345 * np.sin(x1) term in x0_t 
  represents gravity's influence according to flight path angle, showing the highest 
  physical plausibility.
- The last two terms (constant terms) in x1_t are difficult to interpret physically.

#### Term-by-Term Evaluation (Previous Attempt Analysis)
Evaluation results for each term. keep = retain, hold = hold/modify, remove = eliminate:
x0_t:
  - -0.5246 * x0 : HOLD1
  - 0.2037 * x1 : HOLD1
  - -1.2345 * np.sin(x1) : KEEP
  - -1.0312 * 1 : HOLD1
x1_t:
  - 1.9440 * x0 : HOLD1
  - 0.1923 * x1 : HOLD1
  - -12.5970 : HOLD1
  - 9.2669 * 1 : HOLD1

#### Removed Terms List (Ban List)
The following term structures have negatively impacted performance. **Do NOT propose them again**:
None (when no terms were marked as 'remove' in previous iterations)

**Goal:** Reflect the Scientist's insights and guidance in the equation structure.

[Required Conditions (Violation Will Cause Errors)]
1. You can use: import numpy as np
2. Function signature: def [function_name](x0, x1, params): return ...
   - This system is 2-dimensional
   - (Allowed) Available variables: x0, x1 (total 2)
   - (Forbidden) Variables x2 and above are NOT allowed (they don't exist!)
3. params usage: Each function (x0_t, x1_t) has its own independent params array. Start from params[0].
4. params limit: Each term can use at most 10 params.
   - Example: params[0]*x0 (1 used), params[0]*sin(params[1]*x0) (2 used)
5. Meaningful constants: Do NOT use params for physically meaningful constants.
   - Example: params[0]*x/(x+1) is preferred over params[0]*x/(x+params[1])
   - Use direct numbers for constants with clear physical meaning (e.g., 2*np.pi*x0)
6. No duplicates: Equations identical to previous attempts are forbidden. Structural modifications are required.
7. Reasoning required: When proposing each term, provide a physical/mathematical reasoning based on the system description.

[Example (2D System)]
x0_t: ["params[0]*x0", "params[1]*x1"]
x1_t: ["params[0]*x0", "params[1]*np.sin(params[2]*x1)"]
```

---

## 2. Sampler Agent - Output
This is the Sampler's response to the above prompt (based on actual JSON structure).

**Important**: While the prompt requests the `params[]` format, the LLM **flexibly outputs both formats**.

```json
{
  "ode_pairs": [
    {
      "x0_t": [
        {
          "term": "params[0] * np.sin(x1)",
          "reasoning": "Models the influence of gravity proportional to the sine of the flight path angle. Reflects the Scientist's analysis result (KEEP)."
        },
        {
          "term": "params[1] * x0 * np.abs(x0)",
          "reasoning": "Models air resistance (Drag) proportional to the square of velocity. A physically plausible nonlinear term."
        }
      ],
      "x1_t": [
        {
          "term": "params[0] * np.cos(x1)",
          "reasoning": "Represents the cosine component of gravity affecting the flight path angle change."
        },
        {
          "term": "params[1] * x0",
          "reasoning": "Explores the linear influence of velocity on the rate of change of path angle."
        },
        {
          "term": "params[2] * np.sin(x1)",
          "reasoning": "Expresses restoring forces due to the angle itself or complex aerodynamic effects."
        }
      ],
      "pair_reasoning": "Combines the influence of gravity with a resistance model proportional to the square of velocity to capture the physical dynamics of the glider."
    }
  ]
}
```

**Note**: 
- **Prompt instruction**: Recommends using `params[0]*x0` format (example: `["params[0]*x0", "params[1]*x1"]`)
- **Actual LLM output**: Both `params[]` format or specific coefficient format (`-0.4000*x0*x1`) are possible
- **System processing**: Terms without `params[]` are automatically wrapped in `params[N]*(term)` format


---

## 3. Scientist Agent - Input Prompt
This is the Scientist evaluation prompt that includes experiment results (MSE) and ablation study measurements.

```text
You are a senior scientist specializing in ODE (differential equation) discovery.
Analyze the experiment results and perform **term-by-term evaluation**.

**Progress**: Currently on iteration 2 of 10 total

System Description: This experiment is conducted to determine the trajectory and 
dynamic characteristics of a glider... (omitted)

=== Accumulated Insights ===
- Analysis up to this point suggests that the -1.2345 * np.sin(x1) term in x0_t 
  represents gravity's influence according to flight path angle, showing the highest 
  physical plausibility.
- The last two terms (constant terms) in x1_t are difficult to interpret physically.

=== Removed Terms List (marked 'remove' in previous iterations) ===
*Do not propose terms with the same skeleton structure as those below.*
None

=== New Experiment Results ===
[Global Best (Target)] (x0_t: 6.925671e-09, x1_t: 8.836587e-01)
x0_t = -1.0000 * np.sin(x1) - 0.2000 * x0 * np.abs(x0) + 0.0000 * 1
x1_t = -0.5304 * np.cos(x1) + 1.5797 * x0 - 0.0585 * np.sin(x1) - 1.5107 * 1

[Previous Attempt] (x0_t: 7.123456e-09, x1_t: 9.123456e-01)
x0_t = -0.9800 * np.sin(x1) - 0.1800 * x0 * np.abs(x0)
x1_t = -0.5100 * np.cos(x1) + 1.6000 * x0

[Current Attempt] (x0_t: 6.925671e-09, x1_t: 8.836587e-01)
x0_t = -1.0000 * np.sin(x1) - 0.2000 * x0 * np.abs(x0) + 0.0000 * 1
  [Sampler's Reasoning]
  - -1.0000 * np.sin(x1): Models the influence of gravity proportional to the sine of the flight path angle. Reflects the Scientist's analysis result (KEEP).
  - -0.2000 * x0 * np.abs(x0): Models air resistance (Drag) proportional to the square of velocity. A physically plausible nonlinear term.
  - 0.0000 * 1: (reasoning for constant term)

x1_t = -0.5304 * np.cos(x1) + 1.5797 * x0 - 0.0585 * np.sin(x1) - 1.5107 * 1
  [Sampler's Reasoning]
  - ... (reasoning for each term is included here)

[How to Perform Term-by-Term Evaluation]

For all terms in the current attempt, evaluate only the following 2 aspects:

1. Semantic Quality (semantic_quality)
Evaluate based on the system description (desc):
- good: Clearly aligns with the physical/mathematical meaning of the system (**Note: Maximum 3 per function**)
- neutral: Has some relevance but not essential
- bad: Unrelated to or contradicts the system

2. Reasoning (reasoning)
Explain your evaluation in 1-2 sentences. Focus on the system description (physical meaning).

[Important Notes]
- Use coefficient analysis: Terms with coefficients near 0 are removal candidates. If all coefficients are far from 0, optimization may be insufficient.
```

---

## 4. Scientist Agent - Output
This is the actual evaluation result generated by the LLM.

```json
{
  "term_evaluations": {
    "x0_t": [
      {
        "term": "-1.0000 * np.sin(x1)",
        "semantic_quality": "good",
        "reasoning": "A term representing the influence of gravity according to the flight path angle, which is physically plausible."
      },
      {
        "term": "-0.2000 * x0 * np.abs(x0)",
        "semantic_quality": "good",
        "reasoning": "A term proportional to the square of velocity, suitable for modeling air resistance."
      },
      {
        "term": "0.0000 * 1",
        "semantic_quality": "bad",
        "reasoning": "A constant term that has no direct relevance to the dynamic characteristics described in the system description."
      }
    ],
    "x1_t": [
      {
        "term": "-0.5304 * np.cos(x1)",
        "semantic_quality": "good",
        "reasoning": "Reflects the gravity component."
      },
      {
        "term": "1.5797 * x0",
        "semantic_quality": "neutral",
        "reasoning": "Physical basis unclear."
      },
      {
        "term": "-0.0585 * np.sin(x1)",
        "semantic_quality": "neutral",
        "reasoning": "Complex effects but difficult to interpret."
      },
      {
        "term": "-1.5107 * 1",
        "semantic_quality": "bad",
        "reasoning": "Lacks physical meaning."
      }
    ]
  },
  "updated_insight": "The velocity change (x0_t) model has been accurately captured with gravity and quadratic resistance. However, x1_t still requires additional analysis."
}
```

---

## 5. Post-processing: Action Determination Rules

The system tracks/removes each term based on the Scientist's `semantic_quality` response and ablation study results.

### 5.1 Action Decision Logic

**Input:**
- `semantic_quality`: good / neutral / bad
- `impact` (ablation study): positive / neutral / negative

**Rules:**

1. **`semantic_quality == "bad"`** → immediately `remove`
2. **`semantic_quality == "good"` AND `impact == "positive"`** → `keep`
3. **All other cases** → `hold` (start or maintain state tracking)

### 5.2 State Transition Model

State transitions are based on the same **Skeleton (structure with coefficients replaced by C)**:

- **bad → remove** (immediate removal)
- **hold1 → hold2 → remove** (removed after 2 consecutive holds)
- **keep** (maintained, no state change)

### 5.3 Actual Transition Example (Iteration 1 → 2)

| Term (Skeleton) | Iteration 1 | Iteration 2 (Scientist) | Iteration 2 (Impact) | Final Action | State Transition |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `C * np.sin(x1)` (x0_t) | hold1 | good | positive | **keep** | hold1 → keep |
| `C * x0 * abs(x0)` (x0_t) | hold1 | good | positive | **keep** | hold1 → keep |
| `C * 1` (x0_t) | hold1 | bad | - | **remove** | hold1 → remove (immediate) |
| `C * x0` (x1_t) | hold1 | neutral | neutral | **hold2** | hold1 → hold2 |
| `C * np.sin(x1)` (x1_t) | (new proposal) | neutral | neutral | **hold1** | - → hold1 |
| `C * 1` (x1_t) | hold1 | bad | - | **remove** | hold1 → remove (immediate) |

### 5.4 Removed Terms List Accumulation

Skeletons of terms marked as `remove` are accumulated in the **removal list (removed_terms_per_dim)**, and from the next iteration, they are passed to the Sampler in the **"Removed Terms List (Ban List)"** section.

**Format Example:**
```
#### Removed Terms List (Ban List)
The following term structures have negatively impacted performance. **Do NOT propose them again**:
  x0_t: C * 1, C * x1**2
  x1_t: C * 1, C * np.exp(x0)
```

**Notes:**
- The **complete list of removed terms up to now** is passed to the Scientist.
- The **ban list for re-proposal** is passed to the Sampler, and terms with the same skeleton structure cannot be proposed.

---

## 6. Term List to Code Conversion Process

This section shows how the term lists from the Sampler Agent's JSON output are converted into executable Python function code.

### 6.1 Input: LLM JSON Output

The Sampler Agent outputs term lists in JSON format:

```json
{
  "ode_pairs": [
    {
      "x0_t": [
        {"term": "params[0] * np.sin(x1)", "reasoning": "..."},
        {"term": "params[1] * x0 * np.abs(x0)", "reasoning": "..."}
      ],
      "x1_t": [
        {"term": "params[0] * np.cos(x1)", "reasoning": "..."},
        {"term": "params[1] * x0", "reasoning": "..."}
      ]
    }
  ]
}
```

### 6.2 Step-by-Step Conversion Process

**Step 1: Extract Term Strings**

From the JSON, extract term strings for each function:

```python
# For x0_t:
terms = ["params[0] * np.sin(x1)", "params[1] * x0 * np.abs(x0)"]

# For x1_t:
terms = ["params[0] * np.cos(x1)", "params[1] * x0"]
```

**Step 2: Parameter Index Remapping**

The system automatically remaps parameter indices to be consecutive starting from 0:

```python
# Input: terms may have non-consecutive indices like params[5], params[2]
# Output: Remapped to params[0], params[1], etc.
terms, index_mapping = remap_param_indices(terms)
```

**Step 3: NumPy Prefix Auto-fix**

Add `np.` prefix to numpy functions if missing:

```python
# Input: "sin(x1)" or "abs(x0)"
# Output: "np.sin(x1)", "np.abs(x0)"
term = fix_numpy_prefix(term)
```

**Step 4: Auto-wrap Terms Without params[]**

If a term doesn't have `params[]`, wrap it automatically:

```python
# Input: "-0.4000 * x0 * x1"  (no params[])
# Output: "params[0] * (-0.4000 * x0 * x1)"
if not re.search(r'params\[\d+\]', term):
    term = f"params[{next_wrap_idx}]*({term})"
```

**Step 5: Combine Terms with Addition**

All terms are combined with `+` operator:

```python
# Input: ["params[0] * np.sin(x1)", "params[1] * x0 * np.abs(x0)"]
# Combined: "(params[0] * np.sin(x1)) + (params[1] * x0 * np.abs(x0))"
return_expr = " + ".join([f"({term})" for term in terms])
```

**Step 6: Add Constant Term**

A constant term is automatically added at the end:

```python
# After combining all terms, add: + params[N] * 1
# where N is the next available parameter index
return_expr += f" + params[{constant_idx}] * 1"
```

**Step 7: Generate Function Code**

Build the complete Python function:

```python
code = f"""def {func_name}({x_args}, params):
    import numpy as np
    return {return_expr}"""
```

### 6.3 Complete Example

**Input (from LLM JSON):**
```json
{
  "x0_t": [
    {"term": "params[0] * np.sin(x1)", "reasoning": "..."},
    {"term": "params[1] * x0 * np.abs(x0)", "reasoning": "..."}
  ]
}
```

**Extracted Term List:**
```python
terms = ["params[0] * np.sin(x1)", "params[1] * x0 * np.abs(x0)"]
func_name = "x0_t"
dim = 2  # 2D system (x0, x1)
max_params = 10
```

**Processing Steps:**

1. **Remap params** (if needed): Already consecutive, no change
2. **Fix numpy prefix**: Already has `np.`, no change
3. **Check for params[]**: Both terms have params[], no wrapping needed
4. **Find max param index**: `max_param_idx = 1` (from params[1])
5. **Combine terms**: `(params[0] * np.sin(x1)) + (params[1] * x0 * np.abs(x0))`
6. **Add constant**: `+ params[2] * 1`
7. **Generate function signature**: `x0_t(x0, x1, params)`

**Output (Python Function Code):**
```python
def x0_t(x0, x1, params):
    import numpy as np
    return (params[0] * np.sin(x1)) + (params[1] * x0 * np.abs(x0)) + params[2] * 1
```

### 6.4 Special Cases

**Case 1: Term without params[]**

```python
# Input term: "-0.4000 * x0 * x1"
# After auto-wrap: "params[0] * (-0.4000 * x0 * x1)"
# Final code includes: (params[0] * (-0.4000 * x0 * x1))
```

**Case 2: Non-consecutive parameter indices**

```python
# Input: ["params[5] * x0", "params[2] * x1"]
# After remapping: ["params[0] * x0", "params[1] * x1"]
# index_mapping = {5: 0, 2: 1}
```

**Case 3: Missing numpy prefix**

```python
# Input: "sin(x1)"
# After fix: "np.sin(x1)"
```

### 6.5 Code Location

The conversion logic is implemented in:
- **Function**: `terms_to_function_code()` in `utils.py` (lines 220-313)
- **Caller**: `parse_candidates()` in `evolution.py` (lines 418-475)

This conversion ensures that:
1. All LLM outputs are normalized to a consistent format
2. Parameter indices are properly managed
3. NumPy functions are correctly prefixed
4. Terms without parameters are automatically wrapped
5. A constant term is always included for flexibility

---

## 7. Performance Impact Measurement (Ablation Study)

This section explains how the system measures the performance impact of each term using ablation study, which quantifies how removing each term affects the model's accuracy.

### 7.1 Overview

**Purpose**: Determine whether each term in the equation contributes positively, negatively, or neutrally to the model's performance.

**Method**: Ablation study - temporarily remove each term by setting its parameter to zero and measure the change in MSE (Mean Squared Error).

**Key Principle**: 
- If removing a term **increases MSE** → the term is **helpful** (positive impact)
- If removing a term **decreases MSE** → the term is **harmful** (negative impact)
- If removing a term **barely changes MSE** → the term is **neutral**

### 7.2 Step-by-Step Process

**Step 1: Calculate Baseline MSE (All Terms Included)**

First, measure the performance with all terms active:

```python
# Function code (already optimized by BFGS)
# x0_t = params[0]*np.sin(x1) + params[1]*x0*np.abs(x0) + params[2]*1
# Optimized params = [1.0000, 0.2000, 0.0001]

func = make_function_from_code(code, func_name)
x_arrays = get_x_arrays_for_function(func, df_train, params)
y_pred_full = func(*x_arrays)
y_true = df_train[target_col].values

mse_baseline = np.mean((y_pred_full - y_true) ** 2)
# Example: mse_baseline = 6.925671e-09
```

**Step 2: Ablate Each Term (Set Parameter to Zero)**

For each term, create a modified parameter array with that term's parameter set to zero:

```python
def ablation_test_single_term(code_str, func_name, term_idx, params, df, target_col):
    # Create ablated params (copy and set one param to 0)
    params_ablated = params.copy()
    params_ablated[term_idx] = 0.0
    
    # Example: Removing term_idx=0 (params[0]*np.sin(x1))
    # Original: params = [1.0000, 0.2000, 0.0001]
    # Ablated:  params = [0.0000, 0.2000, 0.0001]
    # Result: x0_t = 0.0000*np.sin(x1) + 0.2000*x0*np.abs(x0) + 0.0001*1
    #        = 0.2000*x0*np.abs(x0) + 0.0001*1  (first term removed)
    
    # Execute function with ablated params
    func = make_function_from_code(code_str, func_name)
    x_arrays = get_x_arrays_for_function(func, df, params_ablated)
    y_pred = func(*x_arrays)
    y_true = df[target_col].values
    
    # Calculate MSE without this term
    mse = np.mean((y_pred - y_true) ** 2)
    return mse
```

**Step 3: Calculate Change Rate**

Compare the MSE with and without each term:

```python
for term_idx in range(len(params)):
    # Skip if parameter is already 0 (term not used)
    if abs(params[term_idx]) < 1e-10:
        continue
    
    # Calculate MSE without this term
    mse_without = ablation_test_single_term(
        code, func_name, term_idx, params, df_train, target_col
    )
    
    # Calculate change rate
    # Positive change_rate: MSE increases when term removed → term is helpful
    # Negative change_rate: MSE decreases when term removed → term is harmful
    change_rate = (mse_without - mse_baseline) / (mse_baseline + 1e-10)
```

**Step 4: Classify Impact**

Based on the change rate, classify each term's impact:

```python
threshold = 0.05  # 5% threshold

if change_rate > threshold:
    impact = "positive"  # Term removal makes MSE worse → term is good
elif change_rate < -threshold:
    impact = "negative"  # Term removal makes MSE better → term is bad
else:
    impact = "neutral"  # Little to no change
```

### 7.3 Complete Example

**Input Function:**
```python
# x0_t function (2D system)
def x0_t(x0, x1, params):
    import numpy as np
    return (params[0]*np.sin(x1)) + (params[1]*x0*np.abs(x0)) + params[2]*1

# Optimized parameters (after BFGS optimization)
params = [1.0000, 0.2000, 0.0001]

# Baseline MSE (all terms active)
mse_baseline = 6.925671e-09
```

**Ablation Results:**

| Term Index | Term | params (ablated) | MSE without term | Change Rate | Impact |
|:---|:---|:---|:---|:---|:---|
| 0 | `params[0]*np.sin(x1)` | `[0.0000, 0.2000, 0.0001]` | `8.123456e-09` | +17.3% | **positive** |
| 1 | `params[1]*x0*np.abs(x0)` | `[1.0000, 0.0000, 0.0001]` | `1.234567e-08` | +78.2% | **positive** |
| 2 | `params[2]*1` (constant) | `[1.0000, 0.2000, 0.0000]` | `6.926000e-09` | +0.005% | **neutral** |

**Interpretation:**
- **Term 0** (sin term): Removing it increases MSE by 17.3% → **positive impact** (important term)
- **Term 1** (quadratic term): Removing it increases MSE by 78.2% → **positive impact** (very important term)
- **Term 2** (constant): Removing it barely changes MSE → **neutral impact** (minimal contribution)

**Output Format:**
```python
term_impacts = {
    'x0_t': [
        {
            'term_idx': 0,
            'impact': 'positive',
            'change_rate': 0.173,  # 17.3% increase
            'mse_baseline': 6.925671e-09,
            'mse_without': 8.123456e-09
        },
        {
            'term_idx': 1,
            'impact': 'positive',
            'change_rate': 0.782,  # 78.2% increase
            'mse_baseline': 6.925671e-09,
            'mse_without': 1.234567e-08
        },
        {
            'term_idx': 2,
            'impact': 'neutral',
            'change_rate': 0.00005,  # 0.005% increase
            'mse_baseline': 6.925671e-09,
            'mse_without': 6.926000e-09
        }
    ]
}
```

### 7.4 Integration with Action Determination

The performance impact is combined with semantic quality to determine the final action:

```python
def determine_action(semantic_quality, performance_impact, prev_action):
    # Rule 1: Bad semantic quality → immediate remove
    if semantic_quality == "bad":
        return "remove"
    
    # Rule 2: Good semantic + positive impact → keep
    if semantic_quality == "good" and performance_impact == "positive":
        return "keep"
    
    # Rule 3: All other cases → hold (track state)
    return "hold"  # Will transition: hold1 → hold2 → remove if persists
```

**Example Decision Matrix:**

| Semantic Quality | Performance Impact | Action | Reason |
|:---|:---|:---|:---|
| good | positive | **keep** | Both quality and performance are good |
| good | neutral | **hold** | Good meaning but unclear performance |
| good | negative | **hold** | Good meaning but harmful performance |
| neutral | positive | **hold** | Good performance but unclear meaning |
| neutral | neutral | **hold** | Both unclear |
| neutral | negative | **hold** | Unclear meaning, harmful performance |
| bad | positive | **remove** | Bad meaning regardless of performance |
| bad | neutral | **remove** | Bad meaning regardless of performance |
| bad | negative | **remove** | Bad meaning and harmful performance |

### 7.5 Code Location

The ablation study logic is implemented in:
- **Function**: `calculate_term_performance_impacts()` in `utils.py` (lines 528-617)
- **Helper**: `ablation_test_single_term()` in `utils.py` (lines 492-525)
- **Caller**: `scientist()` in `evolution.py` (lines 682-691)
- **Action Logic**: `determine_action()` in `utils.py` (lines 619-661)

### 7.6 Important Notes

1. **No Re-optimization**: The ablation study does NOT re-optimize parameters. It simply sets one parameter to zero while keeping all others unchanged.

2. **Threshold**: The default threshold is 5% (0.05). Terms with change rates between -5% and +5% are classified as neutral.

3. **Zero Parameters**: Terms with parameters already near zero (|param| < 1e-10) are skipped as they are effectively unused.

4. **Efficiency**: Ablation study is performed on the training data only, not on test sets, to maintain computational efficiency.

5. **Combined Evaluation**: Performance impact alone does not determine the action. It is combined with semantic quality from the Scientist Agent for a holistic evaluation.

---

## 8. Feedback Synthesis: Combining Quantitative and Qualitative Evaluations

This section explains how the system synthesizes quantitative (performance impact) and qualitative (semantic quality) evaluations to generate actionable feedback for the next iteration.

### 8.1 Overview

**Purpose**: Combine two complementary evaluation dimensions to make informed decisions about each term:
- **Quantitative (Performance Impact)**: Measured through ablation study
- **Qualitative (Semantic Quality)**: Assessed by Scientist LLM based on system description

**Output**: Action decisions (`keep`, `hold1`, `hold2`, `remove`) that are synthesized into feedback for the next iteration.

### 8.2 Synthesis Process

**Step 1: Collect Evaluations**

For each term, gather both evaluations:

```python
# From Scientist LLM
semantic_quality = eval_item.get('semantic_quality', 'neutral')  # 'good', 'neutral', 'bad'

# From Ablation Study
impact_info = impact_map.get(term_idx)
performance_impact = impact_info['impact']  # 'positive', 'neutral', 'negative'
```

**Step 2: Extract Term Skeleton**

To track terms across iterations, extract the skeleton (structure with coefficients replaced by 'C'):

```python
def extract_term_skeleton(term: str) -> str:
    """Extract skeleton by replacing coefficients with 'C'."""
    # Input: "-1.2345 * np.sin(x1)"
    # Output: "C*np.sin(x1)"
    
    # Replace params[i] with C
    skeleton = re.sub(r'params\[\d+\]', 'C', term)
    
    # Replace numeric coefficients with C
    skeleton = re.sub(r'-?\d+\.?\d*', 'C', skeleton)
    
    # Remove whitespace for consistent comparison
    skeleton = re.sub(r'\s+', '', skeleton)
    
    return skeleton
```

**Step 3: Retrieve Previous Action**

Track the state of each term across iterations using skeleton mapping:

```python
# Map: skeleton → previous action
skeleton_to_prev_action = {
    "C*np.sin(x1)": "hold1",  # From iteration 1
    "C*x0*abs(x0)": "keep",   # From iteration 1
    "C*1": "remove"            # From iteration 1
}

term_skeleton = extract_term_skeleton(term_str)
prev_action = skeleton_to_prev_action.get(term_skeleton)  # None if new term
```

**Step 4: Determine Action**

Combine all information to determine the action:

```python
def determine_action(semantic_quality, performance_impact, prev_action=None):
    # Rule 1: Bad semantic quality → immediate remove
    if semantic_quality == 'bad':
        return 'remove'
    
    # Rule 2: Good semantic + positive impact → keep
    if semantic_quality == 'good' and performance_impact == 'positive':
        return 'keep'
    
    # Rule 3: State machine for hold cases
    # hold1 → hold2 → remove (after 2 consecutive holds)
    if prev_action == 'hold1':
        return 'hold2'
    if prev_action == 'hold2':
        return 'remove'  # 2 consecutive holds → remove
    
    # Default: start tracking with hold1
    return 'hold1'
```

**Step 5: Update Removal List**

If action is `remove`, add the skeleton to the ban list:

```python
if action == 'remove':
    existing = removed_terms_per_dim.get(func_name, [])
    removed_terms_per_dim[func_name] = merge_remove_list(existing, [term_str])
```

### 8.3 Complete Example: Iteration 1 → 2

**Input: Current Iteration (Iteration 2) Evaluations**

**Function**: `x0_t`
```python
# Current equation
x0_t = -1.0000*np.sin(x1) + -0.2000*x0*abs(x0) + 0.0001*1

# Terms with evaluations
terms = [
    "-1.0000*np.sin(x1)",      # term_idx=0
    "-0.2000*x0*abs(x0)",      # term_idx=1
    "0.0001*1"                 # term_idx=2
]
```

**Step 1: Collect Evaluations**

```python
# Term 0: -1.0000*np.sin(x1)
semantic_quality_0 = "good"        # From Scientist LLM
performance_impact_0 = "positive"  # From ablation (MSE +17.3%)

# Term 1: -0.2000*x0*abs(x0)
semantic_quality_1 = "good"        # From Scientist LLM
performance_impact_1 = "positive"   # From ablation (MSE +78.2%)

# Term 2: 0.0001*1
semantic_quality_2 = "bad"          # From Scientist LLM
performance_impact_2 = "neutral"    # From ablation (MSE +0.005%)
```

**Step 2: Extract Skeletons**

```python
skeleton_0 = extract_term_skeleton("-1.0000*np.sin(x1)")
# Result: "C*np.sin(x1)"

skeleton_1 = extract_term_skeleton("-0.2000*x0*abs(x0)")
# Result: "C*x0*abs(x0)"

skeleton_2 = extract_term_skeleton("0.0001*1")
# Result: "C*1"
```

**Step 3: Retrieve Previous Actions (from Iteration 1)**

```python
skeleton_to_prev_action = {
    "C*np.sin(x1)": "hold1",    # Was hold1 in iteration 1
    "C*x0*abs(x0)": "hold1",    # Was hold1 in iteration 1
    "C*1": "hold1"              # Was hold1 in iteration 1
}
```

**Step 4: Determine Actions**

```python
# Term 0: good + positive + hold1 → keep
action_0 = determine_action("good", "positive", "hold1")
# Result: "keep"

# Term 1: good + positive + hold1 → keep
action_1 = determine_action("good", "positive", "hold1")
# Result: "keep"

# Term 2: bad + neutral + hold1 → remove (bad semantic overrides)
action_2 = determine_action("bad", "neutral", "hold1")
# Result: "remove"
```

**Step 5: Update Removal List**

```python
# Term 2 is removed
removed_terms_per_dim = {
    'x0_t': ['C*1']  # Skeleton added to ban list
}
```

**Step 6: Generate Feedback for Next Iteration**

The system formats the feedback for the Sampler Agent:

```python
feedback = format_scientist_insight_for_prompt(
    insight_list=["Analysis suggests sin(x1) and quadratic terms are essential"],
    removed_terms_per_dim={'x0_t': ['C*1']},
    term_evaluations={
        'x0_t': [
            {'term': '-1.0000*np.sin(x1)', 'action': 'keep'},
            {'term': '-0.2000*x0*abs(x0)', 'action': 'keep'}
            # 'remove' items are excluded (appear in ban list instead)
        ]
    }
)
```

**Output: Feedback Prompt Section**

```text
### SCIENTIST AGENT GUIDANCE
The Scientist agent has analyzed previous experiments and provides the following guidance:

#### Accumulated Knowledge (Theory)
- Analysis suggests sin(x1) and quadratic terms are essential

#### Term-by-Term Evaluation (Previous Attempt Analysis)
Evaluation results for each term. keep = retain, hold = hold/modify, remove = eliminate:
x0_t:
  - -1.0000*np.sin(x1) : KEEP
  - -0.2000*x0*abs(x0) : KEEP

#### Removed Terms List (Ban List)
The following term structures have negatively impacted performance. **Do NOT propose them again**:
  x0_t: C*1
```

### 8.4 State Transition Examples

**Example 1: Hold Progression**

| Iteration | Term | Semantic | Impact | Prev Action | New Action | State Transition |
|:---|:---|:---|:---|:---|:---|:---|
| 1 | `C*x0` | neutral | neutral | - | **hold1** | - → hold1 |
| 2 | `C*x0` | neutral | neutral | hold1 | **hold2** | hold1 → hold2 |
| 3 | `C*x0` | neutral | neutral | hold2 | **remove** | hold2 → remove |

**Example 2: Immediate Keep**

| Iteration | Term | Semantic | Impact | Prev Action | New Action | State Transition |
|:---|:---|:---|:---|:---|:---|:---|
| 1 | `C*np.sin(x1)` | good | positive | - | **keep** | - → keep |
| 2 | `C*np.sin(x1)` | good | positive | keep | **keep** | keep → keep (stable) |

**Example 3: Immediate Remove**

| Iteration | Term | Semantic | Impact | Prev Action | New Action | State Transition |
|:---|:---|:---|:---|:---|:---|:---|
| 1 | `C*1` | bad | neutral | - | **remove** | - → remove (immediate) |

### 8.5 Feedback Format Details

**Term-by-Term Evaluation Format:**

- **KEEP**: Terms that are both semantically good and performance-positive
- **HOLD1**: Terms under observation (first hold)
- **HOLD2**: Terms under observation (second hold, will be removed if held again)
- **REMOVE**: Not shown (appears in ban list instead)

**Removed Terms List Format:**

- Shows skeleton structures (coefficients replaced by 'C')
- Prevents re-proposal of ineffective term structures
- Example: `C*1, C*x1**2, C*np.exp(x0)`

### 8.6 Code Location

The feedback synthesis logic is implemented in:
- **Action Determination**: `determine_action()` in `utils.py` (lines 619-661)
- **Skeleton Extraction**: `extract_term_skeleton()` in `utils.py` (lines 419-456)
- **Feedback Formatting**: `format_scientist_insight_for_prompt()` in `prompt.py` (lines 31-108)
- **Synthesis Orchestration**: `analyze_and_record_observation()` in `evolution.py` (lines 653-837)

### 8.7 Key Design Principles

1. **Deterministic Actions**: Actions are determined by rules, not by LLM, ensuring consistency.

2. **Skeleton-Based Tracking**: Terms are tracked by structure (skeleton), not by exact coefficients, allowing recognition across iterations.

3. **Progressive Removal**: Terms are not immediately removed unless semantically bad. They go through a hold period to avoid premature removal.

4. **Dual Evaluation**: Both quantitative and qualitative aspects are considered, preventing over-reliance on either metric alone.

5. **Feedback Clarity**: The feedback clearly distinguishes between terms to keep, terms to modify, and terms to avoid.
