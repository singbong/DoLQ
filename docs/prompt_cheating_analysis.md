# 프롬프트 치팅 분석 (Prompt Cheating Analysis)

생성일: 2026-01-20

## 요약

**결론: 직접적인 치팅은 없지만, 간접적인 힌트는 존재합니다.**

- ✅ **Ground Truth 값 직접 노출**: 없음
- ✅ **정답 방정식 제공**: 없음
- ⚠️ **이전 시도 결과 제공**: 있음 (MSE + 방정식)
- ⚠️ **간접적 힌트**: 이전 시도의 성공/실패 정보 포함

---

## 1. Ground Truth 사용 위치

### 1.1 최적화 단계 (Optimization)

**위치**: `optimization.py`, `evolution.py`

```python
# use_gt 옵션에 따라 타겟 컬럼 선택
if use_gt:
    target_col = f'x{i}_t_gt'  # Ground truth 사용
else:
    target_col = f'x{i}_t'    # Gradient-based 사용

y_true = self.df_train[target_col].values  # 최적화 타겟으로만 사용
```

**용도**: 
- 파라미터 최적화 시 타겟으로만 사용
- MSE 계산에 사용
- **프롬프트에는 포함되지 않음**

### 1.2 프롬프트 생성

**위치**: `prompt.py`, `evolution.py`

프롬프트 생성 시 ground truth 값은 **전혀 포함되지 않습니다**.

---

## 2. 프롬프트 구성 요소 분석

### 2.1 Sampler 프롬프트 (`make_sampler_ODE_prompt`)

**포함되는 정보**:

1. ✅ **시스템 설명** (`describe`)
   - 물리적/수학적 시스템 설명
   - 변수 의미 설명
   - **치팅 없음**: 일반적인 문제 설명

2. ✅ **Scientist 인사이트**
   - 이전 실험 분석 결과
   - 항 평가 (keep/hold/remove)
   - 제거된 항 목록
   - **치팅 없음**: 이전 시도에서 학습한 정보

3. ⚠️ **이전 시도 결과** (간접적 힌트)
   - 이전 시도의 방정식과 MSE
   - Global Best 방정식과 MSE
   - **간접적 힌트**: 성공한 방정식 구조를 보여줌

4. ✅ **예제**
   - 형식적 예제만 제공
   - **치팅 없음**: 구조만 보여줌

**포함되지 않는 정보**:
- ❌ Ground truth 값 (`x_t_gt` 컬럼 값)
- ❌ 정답 방정식
- ❌ 실제 데이터 포인트 값

### 2.2 Scientist 프롬프트 (`make_analysis_and_record_prompt`)

**포함되는 정보**:

1. ✅ **시스템 설명** (`describe`)
   - **치팅 없음**

2. ⚠️ **실험 결과 비교**
   - Global Best: `(x0_t: 2.107073e-06, x1_t: 2.514873e-02)`
   - Previous Attempt: `(x0_t: 6.682350e-01, x1_t: 2.774116e-02)`
   - Current Attempt: `(x0_t: 6.712514e-01, x1_t: 2.725614e-02)`
   - **간접적 힌트**: MSE 비교를 통해 어떤 방정식이 더 좋은지 알 수 있음

3. ⚠️ **방정식 비교**
   - Global Best 방정식 전체
   - Previous Attempt 방정식 전체
   - Current Attempt 방정식 전체
   - **간접적 힌트**: 성공한 방정식 구조를 볼 수 있음

4. ✅ **Ablation Study 결과**
   - 각 항의 제거 시 영향
   - **치팅 없음**: 실험 결과 기반

**포함되지 않는 정보**:
- ❌ Ground truth 값
- ❌ 정답 방정식 (단, Global Best가 정답에 가까울 수 있음)

---

## 3. 치팅 가능성 평가

### 3.1 직접적 치팅 (Direct Cheating)

| 항목 | 포함 여부 | 평가 |
|------|----------|------|
| Ground truth 값 (`x_t_gt`) | ❌ 없음 | ✅ 치팅 없음 |
| 정답 방정식 직접 제공 | ❌ 없음 | ✅ 치팅 없음 |
| 정답 파라미터 제공 | ❌ 없음 | ✅ 치팅 없음 |

**결론**: 직접적인 치팅은 **없습니다**.

### 3.2 간접적 힌트 (Indirect Hints)

| 항목 | 포함 여부 | 평가 |
|------|----------|------|
| 이전 시도 방정식 | ✅ 있음 | ⚠️ 간접적 힌트 |
| MSE 비교 정보 | ✅ 있음 | ⚠️ 간접적 힌트 |
| Global Best 방정식 | ✅ 있음 | ⚠️ 간접적 힌트 |
| Ablation study 결과 | ✅ 있음 | ✅ 정당한 피드백 |

**분석**:

1. **이전 시도 방정식 제공**
   - **의도**: Evolutionary algorithm의 표준 방식
   - **효과**: 성공한 구조를 참고하여 개선
   - **치팅 여부**: 논문에서 명시하는 경우 정당함
   - **비교**: Genetic Algorithm, CMA-ES 등에서도 사용

2. **MSE 비교 정보**
   - **의도**: 어떤 방정식이 더 좋은지 피드백
   - **효과**: 방향성 제시
   - **치팅 여부**: 일반적인 최적화 피드백

3. **Global Best 방정식**
   - **의도**: 최고 성능 방정식 유지
   - **효과**: 탐색 공간 축소
   - **치팅 여부**: 논문에서 명시 필요

---

## 4. 논문 작성 시 고려사항

### 4.1 명시해야 할 사항

1. **이전 시도 결과 제공**
   - "We provide the LLM with previous attempts and their MSE scores to guide exploration"
   - "The Sampler agent receives feedback from previous iterations"

2. **Global Best 유지**
   - "We maintain the global best solution and provide it as a reference"
   - "The Scientist agent compares current attempts with the global best"

3. **use_gt 옵션**
   - "When `use_gt=True`, we use ground truth derivatives (`x_t_gt`) as optimization targets instead of gradient-based estimates (`x_t`)"
   - "Ground truth values are only used for parameter optimization, not in LLM prompts"

### 4.2 치팅이 아닌 이유

1. **표준 최적화 기법**
   - Evolutionary algorithms에서 일반적
   - 이전 세대 정보 활용은 표준

2. **명시적 피드백**
   - 논문에서 공개적으로 설명 가능
   - 재현 가능

3. **Ground Truth 미노출**
   - 실제 정답 값은 제공하지 않음
   - 방정식 구조만 참고

---

## 5. 개선 제안 (선택사항)

만약 더 엄격한 평가를 원한다면:

### 5.1 Blind Mode 추가

```python
# 옵션: 이전 시도 결과를 숨김
--hide_previous_attempts true
```

### 5.2 제한적 피드백

```python
# 옵션: MSE만 제공, 방정식은 숨김
--feedback_type mse_only
```

### 5.3 Ablation Study

현재 방식과 blind mode를 비교하여:
- 이전 시도 정보의 영향 측정
- 실제로 도움이 되는지 검증

---

## 6. 결론

### 현재 상태

- ✅ **직접적 치팅**: 없음
- ⚠️ **간접적 힌트**: 있음 (이전 시도 결과 제공)
- ✅ **정당성**: Evolutionary algorithm 표준 방식

### 권장사항

1. **논문에서 명시**: 이전 시도 결과를 제공한다고 명시
2. **비교 실험**: Blind mode와 비교하여 영향 측정
3. **투명성**: 프롬프트 예시를 논문에 포함

### 최종 평가

**치팅 수준**: 낮음 (Low)

이전 시도 결과 제공은 일반적인 최적화 기법이며, ground truth 값이나 정답 방정식을 직접 제공하지 않으므로 **치팅으로 간주하기 어렵습니다**. 다만, 논문에서 이를 명시하는 것이 좋습니다.
