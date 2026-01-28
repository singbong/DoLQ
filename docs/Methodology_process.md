# 2. Methodology

**Problem Definition.** 시간에 따른 상태 관측값 $\{(t_i, \mathbf{x}_i)\}_{i=1}^N$ (여기서 $\mathbf{x}_i \in \mathbb{R}^d$)와 시스템에 대한 자연어 설명 *Desc*가 주어졌을 때, 해석 가능한 ODE 시스템 $d\mathbf{x}/dt = \mathbf{f}(\mathbf{x})$를 발견한다. 여기서 $\mathbf{f} = [f_1, \ldots, f_d]^T$이고, 각 $f_j: \mathbb{R}^d \rightarrow \mathbb{R}$는 LLM이 제안한 항들로 구성된 함수이다. 미분값 $\dot{\mathbf{x}}_i$는 데이터 제공 형태에 따라 두 가지 방식으로 준비된다: (1) 상태값 $\mathbf{x}$만 주어진 경우 관측 데이터로부터 수치 미분으로 근사하며, (2) 미분값 $\dot{\mathbf{x}}$도 함께 주어진 경우 제공된 값을 직접 사용한다. 목표는 제안된 함수 $\mathbf{f}$에 대해 (1) 예측 정확도: 손실 $\mathcal{L} = \text{NMSE}(\dot{\mathbf{x}}, \mathbf{f}(\mathbf{x}))$ 최소화, (2) 물리적 해석 가능성: 각 항이 *Desc*의 물리 원리와 일치, 를 만족하는 함수와 계수를 찾는 것이다. 전통적 기호 회귀는 상태값 $\mathbf{x}$를 재구성하지만, 본 방법은 미분값 $\dot{\mathbf{x}}$를 직접 매칭하며 *Desc* 기반 도메인 지식을 활용한다.

**Workflow.** Sampler Agent, Parameter Optimizer, Scientist Agent로 구성된 LLM 기반 반복적 탐색 프레임워크는 다음과 같이 작동한다: (1) Sampler가 *Desc*와 Scientist 피드백을 받아 후보 항들과 근거를 생성, (2) Optimizer가 후보 항들의 계수를 최적화하여 MSE를 산출, (3) Scientist는 결과의 데이터 일치도에 대한 정량적 평가와 *Desc* 기반의 설계 타당성에 대한 정성적 피드백을 종합하여 Sampler에게 전달한다. 이 피드백 루프는 다음 iteration의 탐색 방향을 정교화하는 지침이 된다. 초기화 이후 반복 과정을 통해 물리적 원리에 부합하는 항 조합 탐색과 비효과적 패턴 회피가 이루어진다.

---

### 2.1 Sampler Agent

*Desc*와 Scientist Agent로부터 전달된 과학적 통찰(정성적), 항별 성능 평가 결과(정량적), 그리고 *removed_terms_per_dim*(제거된 항 목록)을 종합하여 각 항마다 물리적/수학적 근거를 포함한 새로운 ODE 후보를 구조화된 형태로 생성하는 LLM 에이전트.

**Structured Output Schema.** $d$차원 시스템에 대해 Sampler는 다음 형식으로 출력한다: `{"ode_pairs": [{"x0_t": [{"term": "params[0]*x0*x1", "reasoning": "..."}], ..., "pair_reasoning": "..."}]}`. 각 `term`은 `params[i]`를 사용한 수학적 표현이며, `reasoning`은 *Desc* 기반 1-2문장 정당화이다. 이는 (1) 추적 가능성: 모든 항이 도메인 지식에 연결된 명시적 근거 보유, (2) 검증: 변수 사용 및 구문 정확성 자동 검증, (3) 일관성: 반복 간 균일한 형식을 보장한다.

**sampler_prompt Configuration.** 완전한 프롬프트는 세 구성 요소로 이루어진다: (1) 시스템 설명 *Desc* (물리적 메커니즘 및 변수 의미), (2) Scientist 피드백 $F_t$ (*Accumulated Insights*, 항별 평가 KEEP/HOLD1, 제거 목록 *removed_terms_per_dim*), (3) 기술적 제약 (변수 범위 $x_0, \ldots, x_{d-1}$, 방정식당 독립 `params` 배열, 항당 최대 $p$개 파라미터, 중복 금지). *removed_terms_per_dim*은 비효과적 항의 구조적 *Skeleton* (계수를 `C`로 대체)을 포함하여 재제안을 방지한다. Soft Forgetting 메커니즘은 확률 $\rho_{forget}$로 *removed_terms_per_dim*을 확률적으로 망각시켜 초기 조기 거부된 항의 재탐색을 허용한다.

---

### 2.2 Parameter Optimization

Differential Evolution으로 전역 최적화를 통해 견고한 초기값을 탐색한 후 BFGS로 정밀한 국소 최적화를 수행하는 2단계 하이브리드 전략.

**generated_code Transformation.** LLM이 제안한 항들의 목록(term list)을 받아서 최적화가 가능한 함수의 형태로 변환한다. 이를 통해 LLM 출력의 변동성에도 불구하고 일관된 수치 최적화 수행이 가능하다.

**Hybrid Optimization.** 기존 연구들에서는 Adam이나 BFGS와 같은 단일 최적화 방식을 주로 사용하였으나, Adam은 정확도가 떨어지고 BFGS는 초기값 설정이 어렵다는 한계가 있다. 이를 극복하기 위해 본 연구는 전역 탐색과 지역 최적화를 순차적으로 수행하는 2단계 하이브리드 전략을 제안한다: (1) *Differential Evolution (DE)* 단계에서는 광범위한 파라미터 공간을 탐색하여 지역해에 빠지지 않는 견고한 초기값을 확보한다. (2) *BFGS* 단계에서는 DE로 얻은 초기값을 바탕으로 준 뉴턴(Quasi-Newton) 방식을 적용하여 정밀한 국소 최적화를 진행한다. 이 접근법은 초기화 민감도를 낮추면서도 높은 수렴 정밀도를 보장한다.


---

### 2.3 Scientist Agent

발견된 수식들에 대한 평가는 *performance_impact*(정량적)와 *semantic_quality*(정성적) 두 가지 측면에서 이루어진다. 기존의 연구들은 주로 수치적인 성능 지표만으로 모델의 우수성을 판단하는 경우가 많았으나, ODE의 각 항이 갖는 의미를 명확히 파악하는 것이 동적 시스템(Dynamic System)의 작동 원리를 깊이 이해하는 데 필수적이라는 본 연구의 가설을 바탕으로 *Desc* 기반의 정성적 평가를 병행한다. 이를 통해 현재 후보 해를 역대 최적 해(Global Best) 및 직전 단계의 해(Previous Generation)와 비교 분석하여 발전 여부를 판단하는 메타 인지 프로세스를 수행하며, 그 결과는 *research_notebook*에 기록된다.

**scientist_analysis_prompt Configuration.** Scientist Agent는 객관적이고 심층적인 판단을 위해 다음 정보를 입력받는다: (1) *Progress* (전체 시행 횟수 대비 현재 iteration), (2) *System Description(Desc)*, (3) *Accumulated Insights* (이전 단계들에서 도출되어 research_notebook에 저장된 누적 지식), (4) *removed_terms_per_dim* (제거된 항 목록), (5) *Experiment Results* (Global Best, Previous Generation, Current Attempt의 NMSE 및 수식), (6) *Sampler's Reasoning* (제안 단계에서의 물리적 정당화).

**performance_impact (Ablation Study).** 각 항의 기여도를 수치적으로 산출하기 위해 개별 파라미터를 0으로 설정하여 분석한다. 모든 항을 포함한 기본 성능(Baseline)과 특정 항을 제거했을 때의 성능 변화율 $\Delta = (\mathcal{L}_{ablated} - \mathcal{L}_{baseline}) / \mathcal{L}_{baseline}$을 비교한다. 변화율에 따라 *performance_impact*를 `positive`($\Delta > \text{threshold}$), `negative`($\Delta < -\text{threshold}$), `neutral`로 분류한다. 이는 전체 방정식 시스템에서 개별 항이 예측 정확도 향상에 실질적으로 기여하는 정도를 정량적으로 파악하여, 각 항의 유효성을 객관적으로 검증하기 위함이다.

**semantic_quality.** Scientist LLM은 시스템 설명(*Desc*)을 바탕으로 각 항의 물리적 타당성을 평가한다. 이는 단순히 수치적인 평가에만 매몰되지 않고, 방정식의 핵심 동작 원리에 대한 이해도를 평가하여 발견된 수식의 과학적 정당성을 확보하기 위한 과정이다. 이를 위해 해당 항이 도메인 지식과 부합하는지 판단하여 세 등급(`good`, `neutral`, `bad`)을 부여한다. 물리적/수학적 의미가 *Desc*와 명확히 일치하는 경우 `good`(방정식당 최대 3개 제한), 관련성은 있으나 필수적이지 않거나 근거가 모호한 경우 `neutral`, 그리고 *Desc*와 무관하거나 물리적 상호작용 원리에 위배되는 경우 `bad`로 분류한다.

**Action Loop and Feedback Synthesis.** 정량적(*performance_impact*) 및 정성적(*semantic_quality*) 평가 결과를 결합하여 각 항의 최종 action(`keep`, `hold`, `remove`)을 결정한다.
- **Action Rules**: *semantic_quality*가 *bad*인 경우 즉시 `remove`하고, *good*이면서 *positive*인 경우만 `keep`으로 유지한다. 그 외의 모든 조합은 `hold` 상태로 관리하며, `hold`를 연속으로 2회 이상 부여받은 항은 시스템에 유의미한 영향이나 통찰을 주지 못하는 것으로 판단하여 최종적으로 `remove`로 전환한다.
- **removed_terms_per_dim Management**: `remove`로 결정된 항들은 *removed_terms_per_dim*에 추가된다. 이때 항들은 계수를 `C`로 치환한 형태인 *Normalized Structure*로 저장된다. 이를 통해 동일한 물리적 의미를 가진 수식 구조가 중복 제안되는 것을 구조적으로 원천 차단한다. 또한, 최적화 미흡 등으로 인해 유효한 항이 초기에 성급하게 배제되었을 가능성을 고려하여 *Soft Forgetting* 메커니즘을 적용한다. 이는 일정 확률($\rho_{forget}$)로 *removed_terms_per_dim*에서 특정 항을 일시적으로 제외함으로써, 시스템이 고도화된 후 해당 항의 잠재적 가치를 다시 검증할 수 있는 기회를 제공한다.
- **updated_insight Generation**: 위 평가 결과와 함께 새롭게 도출된 과학적 통찰인 *updated_insight*를 생성하여 *research_notebook*을 업데이트하고, 이를 다음 반복의 Sampler에게 전달하여 탐색을 유도한다.