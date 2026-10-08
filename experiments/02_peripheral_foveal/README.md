# Experiment 02: Peripheral + Foveal Vision

## 목적

이 실험은 다음 질문을 검증합니다.

> 전체 장면을 항상 유지한 상태에서, 필요한 위치만 고해상도로 추가로 읽는 방식이 실제로 도움이 되는가?

핵심은 정보를 제거하는 것이 아니라, 전체 장면을 유지하면서 "선택적 고해상도 readout"을 추가하는 것입니다.

---

## 핵심 아이디어

기존 실험의 구조와 달리, 이 실험은 다음을 전제로 합니다.

1. 전체 112x112 캔버스를 절대 가리지 않는다.
2. Peripheral vision은 전체 장면을 저해상도로 요약한다.
3. Foveal vision은 예측된 위치 주변을 고해상도로 읽는다.
4. 최종 분류는 전체 정보 + foveal 정보 + focus 좌표를 종합한다.
5. 따라서 실험의 질문은 "정보를 없앤 뒤 맞히는가?"가 아니라
   "전체 정보를 가진 상태에서 추가된 고해상도 정보가 유익한가?"이다.

---

## 모델 구조

### 1) Peripheral Path
- 전체 캔버스를 112x112 형태로 유지
- 낮은 해상도 feature map 생성
- saliency map에서 "어디를 더 자세히 볼지" 예측

### 2) Foveal Path
- predicted focus 위치 주변만 crop
- 해당 patch를 고해상도로 처리
- local detail embedding 생성

### 3) Combined Decision
최종 분류는 다음을 결합합니다.

- peripheral global feature
- foveal local feature
- predicted focus coordinates

즉,

```
Global Scene + Local Detail + Focus Location -> Classifier
```

형태가 됩니다.

---

## 왜 중요한가

이 구조는 인간 시각처럼 실제로 동작하는 형태와 더 가깝습니다.

- 인간은 전체 장면을 동시에 보는 것이 아니라, 
  넓은 시야를 유지하면서도 필요한 영역을 더 자세히 보는 방식입니다.
- 그렇기 때문에 single hard cropping 구조보다 훨씬 현실적입니다.
- 이 실험은 "semantic context를 선택적으로 추가로 읽는 구조"를 평가합니다.

---

## 비교 대상

### A. Full-only baseline
- 전체 화면만 보고 분류
- 가장 단순한 기준선

### B. Peripheral + Foveal
- 전체 장면을 유지하고, 선택적 고해상도 patch를 추가로 읽음
- 핵심 실험 모델

### C. Oracle Foveal
- 실제 위치를 알고 있는 모델
- upper bound로서 의미

---

## 실험 수치

실험은 다음 지표를 기본으로 봅니다.

- Classification Accuracy
- Fovea center prediction error
- Saliency map / focus trajectory visualization

---

## 실행 방법

```bash
python experiments/02_peripheral_foveal/run_experiment.py
```

실행 시 생성되는 결과물:

- `experiments/02_peripheral_foveal/experiment02_saccade_result.png`
- `experiments/02_peripheral_foveal/experiment02_peripheral_foveal.pt`

---

## 해석

### 기대되는 결과

- Full-only baseline이 어느 정도는 잘 맞을 수 있음
- Peripheral + Foveal 모델은 전체 시야를 유지하면서도 local detail을 이용해 정확도를 상승시킬 수 있음
- Oracle foveal은 upper bound를 제공하여, learned focus의 잠재적 한계를 보여줌

### 핵심 결론

이 실험은 다음 가설을 검증합니다.

> 선택적 고해상도 읽기는 전체 정보가 유지된 상태에서 의미 있는 보조 정보가 될 수 있다.

이는 Exp 01의 "정보를 가리는 구조"와 대조적으로,
"전체를 본다는 사실을 유지하면서 필요한 부분만 스스로 더 자세히 읽는 구조"를 평가하는 실험입니다.

---

## 위치와 연결

- Exp 01: `experiments/01_semantic_clustering/`
- Exp 02: `experiments/02_peripheral_foveal/`

Exp 02는 Exp 01의 보완 실험으로, 실제 시각 시스템에 더 가까운 구조를 시도합니다.

---

## 핵심 메시지

Exp 01은 "어디를 보아야 하는가"를 학습하는 문제를 다룹니다.
Exp 02는 "전체 정보를 유지한 채로, 필요한 부분만 더 자세히 보는 방식"이 실제로 유용한지를 검증합니다.

이 둘을 함께 보면, semantic context 방식이 단순히 input을 잘라서 보는 구조가 아니라,
전체 장면을 유지하면서 필요한 정보를 점진적으로 보강하는 방식으로 확장될 수 있음을 보여줍니다.
