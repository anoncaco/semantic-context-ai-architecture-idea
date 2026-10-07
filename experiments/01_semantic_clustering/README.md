# Semantic Context AI Architecture - Experiment Progress

## 📋 Project Status

This repository contains research and implementation of a **Semantic Context-based AI Architecture** that differs fundamentally from token-based language models.

### Core Hypothesis
> AI systems should process information through **unified semantic fields** where attention is **adaptively guided by information content**, and reasoning happens through **recursive refinement with persistent state**, rather than sequential token generation.

---

## 🧪 Experiment 01: Adaptive Foveated Attention with Uncertainty-Guided Search

### Overview

We test the core architectural hypothesis using a visual foveation model on synthetic MNIST-like digits.

**Key Question:** Can a model learn to adaptively direct its focus toward information-rich regions, accumulating evidence through recurrent state, and make decisions based on learned uncertainty?

### Implementation Versions

#### Version A: Discrete Foveation (`run_experiment.py`)
- **Approach:** Hard overlay of focal patch on blurred background
- **Focus Movement:** Discrete argmax on information map
- **Decision:** Hard threshold on confidence
- **Status:** Baseline (non-differentiable focus selection)

#### Version B: Fully Differentiable Foveation (`run_experiment_diff.py`) ⭐ **RECOMMENDED**
- **Approach:** 2D Gaussian mask for soft foveation blending
- **Focus Movement:** Spatial softmax for expected coordinates
- **Decision:** Soft gating based on confidence
- **Status:** Fully differentiable, end-to-end learnable
- **Advantage:** Gradients flow through entire pipeline, attention optimized by loss

### Architecture

```
Input Image (28×28)
    ↓
[Foveated Image Generator]
    - Gaussian-blended focus region + blurred periphery
    - Differentiable w.r.t. focus coordinates
    ↓
[Full-Canvas Encoder]
    - Conv2D over entire 28×28 image
    - Preserves spatial structure
    - Outputs: feature map (14×14) + saliency map
    ↓
[Information Map Analysis]
    - Compute spatial softmax of saliency
    - Expected next focus point (continuous coordinates)
    - Fully differentiable
    ↓
[Recurrent State (GRU)]
    - Accumulates information across glimpses
    - Provides context for next decision
    ↓
[Uncertainty-Driven Decision]
    - Soft gating: move focus only if uncertain
    - Gate = (confidence < threshold)
    - Smooth transitions, no hard switches
    ↓
[Repeat for K glimpses]
    - Typically 3 glimpses for MNIST
    ↓
[Final Classification]
    - Output: 10-class logits
```

### Key Technical Innovations

#### 1. **Differentiable Foveation Blending**
```python
# Gaussian mask centered at continuous focus point
mask = exp(-(dist² / 2σ²))

# Soft blending (not discrete overlay)
foveated = mask * original + (1-mask) * blurred
```
✅ Gradients flow through focus coordinates
✅ No discrete indexing breaks gradient flow

#### 2. **Spatial Softmax for Expected Focus**
```python
# Normalize saliency to probability
probs = softmax(info_map)

# Expected coordinates (like computing mean)
exp_y = Σ(y_coord * probs[y,x])
exp_x = Σ(x_coord * probs[y,x])
```
✅ Smooth gradient field
✅ Unlike argmax, doesn't lose gradient information

#### 3. **Soft Gating for Adaptive Movement**
```python
# Confidence-based soft gate
gate = (confidence < threshold).float()

# Smooth focus transition
focus = gate * next_focus + (1-gate) * focus
```
✅ No hard decisions
✅ Confidence uncertainty directly influences movement

#### 4. **End-to-End Differentiability**
- Entire pipeline is differentiable
- Focus movement optimized by classification loss
- Attention mechanism learns from gradients
- No two-stage separation

### Measurements & Diagnostics

#### 1. Classification Accuracy
- **Metric:** Test accuracy on synthetic MNIST
- **Expected:** 85-95% with 3 glimpses
- **Interpretation:** High accuracy validates adaptive focus usefulness

#### 2. Feature Clustering
- **Metric:** Intra-class vs inter-class distance in final GRU state
- **Expected:** inter/intra ratio > 2.0
- **Interpretation:** GRU state learns digit-specific representations

#### 3. Focal Trajectories
- **Metric:** Visualize focus point movement across glimpses
- **Expected:** 
  - Glimpse 1: Center (initialization)
  - Glimpse 2-3: Move toward discriminative strokes/loops
  - Different digits → different trajectories
- **Interpretation:** Model learns where diagnostic information resides

#### 4. Saliency Map Evolution
- **Metric:** Information map (saliency) changes across glimpses
- **Expected:** 
  - Early: broad, diffuse attention
  - Later: concentrated on digit-specific features
- **Interpretation:** Uncertainty-driven refinement working

### Expected Results

#### Scenario A: Strong Success ✅✅✅
- Accuracy: >90%
- Separation ratio (inter/intra): >2.5x
- Focal trajectories: Clear digit-specific patterns
- Saliency maps: Progressive concentration

**Conclusion:** Adaptive foveated attention successfully learns where to look

#### Scenario B: Moderate Success ✅✅
- Accuracy: 80-90%
- Separation ratio: 1.5-2.5x
- Focal trajectories: Noisy but trending toward features
- Saliency maps: Some structure visible

**Conclusion:** Mechanism working but needs refinement (higher capacity, better hyperparameters)

#### Scenario C: Limited Success ✅
- Accuracy: 70-80%
- Separation ratio: <1.5x
- Focal trajectories: Random or uniform
- Saliency maps: No clear structure

**Conclusion:** Needs architectural changes (larger model, different loss weighting)

#### Scenario D: Failure ❌
- Accuracy: <70%
- Training loss not decreasing
- No meaningful focal trajectories

**Conclusion:** Training issue or fundamental design problem

### Running the Experiments

#### Prerequisites
```bash
pip install torch numpy matplotlib scipy
```

#### Version A (Discrete)
```bash
python experiments/01_semantic_clustering/run_experiment.py
```
**Output:**
- `feature_clustering.png` - PCA visualization of final GRU states
- `focal_trajectories.png` - Focal point paths for each digit
- Console: test accuracy + clustering metrics

#### Version B (Differentiable) ⭐ **RECOMMENDED**
```bash
python experiments/01_semantic_clustering/run_experiment_diff.py
```
**Output:**
- `feature_clustering_diff.png` - Better clustering expected
- `focal_trajectories_diff.png` - Smoother trajectories
- Console: test accuracy + clustering metrics + gradient flow confirmation

### Interpretation Guide

#### Feature Clustering Visualization
- **Good:** 10 distinct colored clusters, minimal overlap
- **Acceptable:** Loose clusters with some cross-digit overlap
- **Poor:** Random scatter with no visible structure
- **Debug:** If poor, check:
  - Is model training? (loss decreasing?)
  - Is GRU state accumulating? (state variance across glimpses?)
  - Is information map learning? (saliency structure visible?)

#### Focal Trajectories
- **Good:** 
  - Red → Yellow → Green progression
  - Moves toward digit-specific features
  - Different patterns per digit
- **Poor:**
  - All images same trajectory
  - Random movement
  - Stuck at center

#### Comparison: Version A vs Version B
- **Version A accuracy:** Baseline
- **Version B accuracy:** Expected to be better or similar (due to better optimization)
- **Key difference:** Version B should have smoother, more consistent trajectories

---

## 🔗 Theoretical Connections

| Architecture Concept | Experiment Implementation |
|---------------------|----------------------------|
| **Unified semantic field** | Single foveated canvas (not split streams) |
| **Adaptive information seeking** | Focus moves toward high-saliency regions |
| **Full spatial encoding** | Conv2D over entire 28×28 image |
| **Uncertainty-driven decisions** | Soft gating based on confidence |
| **Recursive refinement** | $Z_0 \to Z_1 \to Z_2 \to Z_3$ via GRU |
| **Persistent state** | GRU accumulates context across glimpses |
| **Differentiable attention** | Entire pipeline backprop-able |
| **End-to-end optimization** | Loss directly optimizes focus movement |

---

## 📊 Next Steps

### Immediate (If Successful)
1. ✅ Reproduce results from both versions
2. ✅ Compare accuracy, clustering quality, trajectory smoothness
3. ✅ Analyze which features GRU state learns
4. ✅ Ablation: remove GRU, test on single glimpse

### Short-term
1. Test on real MNIST dataset
2. Extend to CIFAR-10 (more complex visual structure)
3. Compare against baselines:
   - Single-glimpse CNN
   - Fixed multi-glimpse (no adaptive movement)
   - Random focal points
4. Investigate what information map learns

### Medium-term
1. Implement proper Recurrent Latent Thinking ($Z_n \to Z_{n+1}$ refinement)
2. Add halting head for autonomous glimpse count
3. Integrate semantic memory paging (from README.md architecture)
4. Test on real datasets at scale

### Long-term
1. Combine with language modeling
2. Multi-modal fusion (image + text)
3. Hardware efficiency analysis (Latency budget, power consumption)
4. Compare to modern attention mechanisms (Transformers, Mamba, etc.)

---

## 📚 Core References

- **Vision Transformers (ViT):** Dosovitski et al., 2020 - patch-based visual processing
- **Recurrent Attention Networks:** Mnih et al., 2014 - spatial attention mechanisms
- **Spatial Softmax:** Jaderberg et al., 2015 - differentiable spatial pooling
- **Gumbel-Softmax:** Maddison et al., 2017 - differentiable discrete sampling
- **Transformer Attention:** Vaswani et al., 2017 - foundation for modern attention

---

## 🏗️ Repository Structure

```
.
├── README.md                           (Main architecture proposal)
├── experiments/
│   └── 01_semantic_clustering/
│       ├── README.md                   (This file - Experiment 01 guide)
│       ├── run_experiment.py           (Version A: Discrete foveation)
│       ├── run_experiment_diff.py      (Version B: Differentiable foveation) ⭐
│       ├── mnist_selective_foveated.py (Keras version - archived)
│       ├── feature_clustering.png      (Output from Version A)
│       ├── feature_clustering_diff.png (Output from Version B)
│       ├── focal_trajectories.png      (Output from Version A)
│       └── focal_trajectories_diff.png (Output from Version B)
```

---

**Status:** Experiment 01 fully implemented and ready for execution
**Last Updated:** 2026-10-07
**Version:** B (Fully Differentiable)
