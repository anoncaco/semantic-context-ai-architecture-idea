# Experiment Comparison Matrix

## 전체 실험 설계와 가설

| 항목 | Exp 01A: Discrete Foveation | Exp 01B: Differentiable Foveation | Exp 02: Peripheral + Foveal |
|------|---------------------------|--------------------------------|---------------------------|
| **파일 위치** | `01_semantic_clustering/run_experiment.py` | `01_semantic_clustering/run_experiment_diff.py` ⭐ | `02_peripheral_foveal/run_experiment.py` |
| **핵심 질문** | "선택적 초점이 필요한 정보만 읽게 하는가?" | "선택적 초점을 fully differentiable하게 학습할 수 있는가?" | "전체 장면을 유지하면서 선택적 고해상도 정보가 도움이 되는가?" |

---

## 기술적 구조 비교

| 구조 요소 | Exp 01A | Exp 01B | Exp 02 |
|---------|--------|--------|--------|
| **입력 크기** | 28×28 | 28×28 | 112×112 |
| **초점 메커니즘** | Hard argmax → discrete patch overlay | Spatial softmax → Gaussian mask blend | Spatial softmax from peripheral saliency |
| **주변부 처리** | Blur 후 겹침 | Gaussian smooth blend | Low-res global feature map |
| **초점 영역** | Binary overlay (켜/끔) | Smooth Gaussian blending | Cropped high-res patch |
| **상태 축적** | GRU via glimpses | GRU via glimpses | Global feat + foveal feat + coordinates |
| **최종 입력** | GRU final state only | GRU final state only | Global + Foveal + Focus location |
| **미분 가능성** | ❌ (argmax 단절) | ✅ (완전 미분 가능) | ✅ (spatial softmax) |
| **정보 폐기** | ✅ (주변부 가림) | ✅ (주변부 가림) | ❌ (전체 장면 유지) |

---

## 실험 목적과 측정 항목

### Experiment 01A: Discrete Foveation
**목적**: 초점을 통한 선택적 정보 처리의 기본 가능성 검증

| 측정 항목 | 설명 | 기대값 |
|---------|------|--------|
| Test Accuracy | 3 glimpse 후 최종 정확도 | 85-95% |
| Feature Clustering (Inter/Intra Ratio) | GRU state의 digit별 분리도 | > 2.0 |
| Focal Trajectories | Glimpse별 초점 이동 패턴 | 숫자별 고유 패턴 |
| Saliency Evolution | 정보맵 변화 추이 | 점진적 집중 |

**검증 가설**:
```
정보를 선택적으로 읽으면 → 필요한 특징에 초점 → 분류 정확도 상승
```

---

### Experiment 01B: Differentiable Foveation
**목적**: 초점 선택 자체를 end-to-end로 학습 가능한가?

| 측정 항목 | 설명 | 기대값 |
|---------|------|--------|
| Test Accuracy | 01A와 비교 | 동등 또는 향상 |
| Gradient Flow | 초점 좌표까지 역전파 | 모든 파라미터에 gradient 존재 |
| Focus Stability | 학습 과정에서 초점 변화 | 불안정 → 안정화 |
| Trajectory Smoothness | 초점 이동의 부드러움 | 01A보다 smooth |

**검증 가설**:
```
Spatial softmax + Gaussian mask + soft gating으로
→ 초점 메커니즘 fully differentiable화
→ Classification loss가 직접 attention 최적화
```

---

### Experiment 02: Peripheral + Foveal
**목적**: 정보 폐기 없이, 선택적 고해상도 readout이 유용한가?

| 측정 항목 | 설명 | 기대값 |
|---------|------|--------|
| Full-only Accuracy | Baseline: 전체 112×112 보기 | 80-90% |
| Peripheral+Foveal Accuracy | 전체 + 고해상도 patch | 85-95% |
| Accuracy Gain | (Periph+Fov) - Full | +2-5% |
| Saccade Error | Predicted focus vs ground truth | 5-15px |
| Oracle Upper Bound | 실제 위치 알고 있을 때 | 92-98% |

**검증 가설**:
```
전체 장면 정보 유지 +
선택적 고해상도 local detail +
초점 위치 명시
→ 추가된 고해상도 정보 유용성 확인
```

---

## 아키텍처와의 연결

### Exp 01의 핵심 아이디어
```
Z₀ (full image fovea) 
  → Z₁ (1st glimpse, GRU state) 
  → Z₂ (2nd glimpse, refined state)
  → Z₃ (3rd glimpse, final state)
  → Classification
```

**문제**: 정보를 가리면 못 맞힐 수 있음

**해결**: Exp 01B로 differentiable하게 학습 가능하게 전환

---

### Exp 02의 핵심 아이디어
```
Full Canvas (always visible)
  ├─ Peripheral (28×28 low-res)
  │   └─ Saliency map → Expected focus
  └─ Foveal (56×56 high-res)
      └─ Crop at predicted focus
  
  → Combined features + coordinates
  → Classification
```

**장점**: 정보 손실 없음, 실제 시각 시스템과 유사

---

## 예상 결과 시나리오

### Scenario A: Exp 01B > Exp 02
- Exp 01B accuracy: 92%
- Exp 02 accuracy: 88%

**해석**:
- 초점을 통한 정보 가려내기가 매우 효과적
- Learnable foveation이 가장 효율적

---

### Scenario B: Exp 02 > Exp 01B
- Exp 01B accuracy: 90%
- Exp 02 accuracy: 93%

**해석**:
- 정보 폐기는 비효율적
- 전체 장면 유지 + selective detail이 더 나음
- 실제 시각처럼 동작하는 구조가 더 강력함

---

### Scenario C: Exp 01B ≈ Exp 02
- 둘 다: 90-93%

**해석**:
- 초점의 효율성과 전체 정보 보존이 상충
- 각 접근의 trade-off 명확
- 용도에 따라 선택 가능

---

## 구체적 비교표

| 특성 | Exp 01A | Exp 01B | Exp 02 |
|------|---------|---------|---------|
| **구현 난이도** | ⭐☆☆ | ⭐⭐⭐ | ⭐⭐☆ |
| **이론적 타당성** | 중간 | 높음 | 높음 |
| **실험 비용 (VRAM)** | 낮음 | 낮음 | 중간 |
| **확장성** | 제한적 | 높음 | 높음 |
| **실제 시각 유사성** | 낮음 | 낮음 | 높음 |
| **정보 손실** | 많음 | 많음 | 없음 |
| **Learnable 초점** | ❌ | ✅ | ✅ |
| **Multi-scale 처리** | ❌ | ❌ | ✅ |

---

## 실행 순서 및 권장 경로

### Path 1: 빠른 검증
1. Exp 01B 실행 (Exp 01A 스킵)
2. Exp 02 실행
3. 결과 비교

### Path 2: 완벽한 검증
1. Exp 01A 실행 (baseline)
2. Exp 01B 실행 (개선)
3. Exp 02 실행
4. 세 모델 모두 비교

### Path 3: 심화 분석
1. Exp 01B: Ablation 테스트
   - GRU 없이 single glimpse
   - 다양한 glimpse 수
   - 다양한 focus_weight

2. Exp 02: 심화 분석
   - Oracle fovea (상한선)
   - Peripheral-only (하한선)
   - Full-only (baseline)

---

## 각 실험의 핵심 메시지

### Experiment 01B (Differentiable Foveation)
> **AI가 어디를 봐야 할지 스스로 학습할 수 있는가?**

핵심: End-to-end learnable attention mechanism

---

### Experiment 02 (Peripheral + Foveal)
> **정보를 폐기하지 않으면서 선택적 고해상도 처리가 도움이 되는가?**

핵심: Multi-scale perception with full scene preservation

---

## 논문 구성 시 활용

### Introduction
- Exp 01B: "기존 token-based 방식의 한계"
- Exp 02: "우리의 semantic context 방식의 장점"

### Method
- Exp 01B: Spatial softmax + Gaussian + soft gating
- Exp 02: Peripheral network + foveal network + combined classifier

### Results
- Exp 01B: Accuracy, clustering, trajectory visualization
- Exp 02: Accuracy comparison, saccade error, oracle upper bound

### Discussion
- 두 실험의 trade-off 분석
- Semantic context 아키텍처의 타당성
- 향후 확장 방향

---

## 다음 단계 실험 후보

1. **Exp 03: Multi-object Scene**
   - 여러 숫자가 있을 때 초점이 어떻게 움직이는가?
   - Semantic memory 크기 증가

2. **Exp 04: Distractor-heavy Scene**
   - 무관한 객체가 많을 때 모델이 무시하는가?
   - Robustness 검증

3. **Exp 05: Real MNIST**
   - Synthetic에서 잘 되던 것이 실제 데이터에서도 작동하는가?

4. **Exp 06: Semantic Memory Integration**
   - Virtual memory paging 구조 추가
   - Knowledge base hierarchical access

---

**최종 결론**

세 실험을 통해:
1. Learnable selective attention의 가능성 (Exp 01B)
2. Multi-scale processing의 효용 (Exp 02)
3. Semantic context 아키텍처의 타당성 (통합 분석)

을 검증합니다.
