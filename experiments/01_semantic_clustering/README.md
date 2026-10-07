# Experiment 01: Semantic Clustering in Shared Latent Space

## Overview

This experiment tests the **core hypothesis** of the semantic context architecture:

> **Can semantically similar visual features naturally cluster in a shared latent space when all patches are encoded with identical weights, without explicit supervision on visual concepts?**

---

## Motivation

Current language models process information sequentially (token-by-token) with autoregressive generation. The proposed architecture hypothesizes that by:

1. Forcing all input features (patches) through **one shared encoder**
2. Mapping them to a **common latent space**
3. Allowing the model to **selectively attend** to relevant features
4. **Accumulating state recursively**

...the model can learn meaningful *semantic concepts* without being told what those concepts are.

**MNIST provides a clean testbed:**
- Simple visual structure (digits 0-9)
- Clear semantic structure (curves, strokes, closures, symmetries)
- Small enough to visualize and interpret embeddings

**Key insight:** We're not trying to predict the digit label. We're testing whether patches with similar visual *structure* (e.g., curved strokes in both '0' and '8') naturally land in similar regions of the latent space.

---

## Hypothesis

### Main Hypothesis (H1)
**Shared encoding induces semantic clustering.**

If all patches use identical weights, then:
- Patches with similar visual patterns should have similar embeddings
- Patches from the same digit should form coherent clusters in latent space
- This clustering should emerge **without explicit supervision on visual concepts**

### Secondary Hypothesis (H2)
**Selective attention learns meaningful sampling patterns.**

Given the learned latent space, a recurrent attention mechanism should:
- Learn which patches are diagnostic for digit classification
- Attend to digit-specific structures (e.g., loops for '0', vertical stroke for '1')
- Accumulate information efficiently across glimpses

### Tertiary Hypothesis (H3)
**Related digits show similar embedding distributions.**

Digits with shared visual features (e.g., '0', '6', '8', '9' all have loops) should:
- Have similar mean embeddings
- Show high cosine similarity between digit-class centroids
- Have overlapping or adjacent clusters in t-SNE space

---

## Experimental Design

### Architecture

```
Input Image (28×28)
    ↓
[Patch Extractor]
    ↓ (extract_patches: 7×7, stride=4)
    ↓
Candidate Patches (36 patches per image)
    ↓
[Shared Conv2D Encoder] ← KEY CONSTRAINT: identical weights for all patches
    ↓
Patch Embeddings (36 × 64-d vectors)
    ↓
[Selective Attention Module] (3 glimpses)
    ↓ (each glimpse: score → attend → update state)
    ↓
Recurrent State (64-d)
    ↓
[Classifier Head]
    ↓
Digit Logits (10 classes)
```

### Key Components

#### 1. Patch Extraction (Candidate Generation)
- **Input:** 28×28 grayscale image
- **Operation:** Sliding window with patch_size=7, stride=4
- **Output:** 6×6 grid of 7×7 patches = 36 patches per image
- **Purpose:** Create multiple candidate regions that may contain semantic content

#### 2. Shared Conv2D Encoder (Semantic Mapping)
- **Architecture:**
  ```
  Conv2D(16, kernel_size=3, padding='same', relu)
    → GlobalAveragePooling2D()
    → Dense(64)
  ```
- **Key constraint:** All 36 patches use **identical weights**
- **Input:** [batch, 36, 7, 7, 1]
- **Output:** [batch, 36, 64] — 36 patch embeddings in shared latent space
- **Purpose:** Force all patches into common coordinate system

#### 3. Selective Attention (Glimpse Module)
- **Mechanism:** Scaled dot-product attention with learnable query/key/value projections
  ```
  score(patch_j) = softmax( score_fn(tanh(key(patch_j) + query(state))) )
  glimpse = weighted_sum( score(patch_j) * patch_j )
  new_state = GRU( feature_to_hidden(glimpse), old_state )
  ```
- **Iterations:** 3 glimpses (learnable halting point)
- **Purpose:** Learn which patches are informative; accumulate information

#### 4. Classifier Head
- **Input:** Final recurrent state (64-d)
- **Output:** 10 logits for digits 0-9

---

## Experimental Setup

### Hyperparameters

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Patch size | 7×7 | Captures local structure (edge, corner, stroke) |
| Stride | 4 | Overlap ensures coverage; small gap (3 pixels) |
| Latent dim | 64 | Sufficient capacity for digit features |
| Hidden dim (GRU) | 64 | Match latent space dimension |
| Glimpses | 3 | Empirical sweet spot (1=too greedy, 5=too slow) |
| Batch size | 128 | Standard MNIST training |
| Epochs | 8 | Quick convergence (~2 min on GPU) |
| Optimizer | Adam(lr=1e-3) | Standard deep learning default |

### Data

- **Train:** 60,000 MNIST images (90% train, 10% val)
- **Test:** 10,000 MNIST images
- **Preprocessing:** Normalize to [0, 1]

---

## Measurements & Diagnostics

### 1. Classification Accuracy

**What:** Standard test accuracy on held-out MNIST test set

**Expected:** ~95-97% (selective attention should work well)

**Interpretation:**
- High accuracy (>95%): Selective attention learned meaningful patterns
- Low accuracy (<90%): Either training issue or architecture mismatch

### 2. Attention Pattern Visualization

**What:** For 8 random test images, visualize which patches the model attends to at each glimpse

**Method:**
- Generate heatmaps showing attention weights for each of 3 glimpses
- Overlay on original image with scatter points at patch centers
- Highlight most-attended patch (argmax) with rectangle

**Expected:**
- Glimpse 1: Broad exploration (multiple patches with non-zero attention)
- Glimpse 2-3: Convergence on specific regions
- Different digits attend to different regions (e.g., '0' attends to boundary, '1' attends to vertical stroke)

**Interpretation:**
- If attention focuses on "meaningless" regions → encoder not learning structure
- If attention focuses on digit-specific features → selective attention working

**Output file:** `attention_glimpses.png`

### 3. Patch Embedding t-SNE Visualization

**What:** Dimensionality reduction of all patch embeddings to 2D

**Method:**
1. Extract patch embeddings from 6000 test images: ~216,000 patches total
2. Apply t-SNE (perplexity=30, 2D)
3. Color each point by its source digit (0-9, different colors)

**Expected:**
- 10 distinct clusters (one per digit)
- Minimal overlap between clusters
- Some clusters closer (similar features): '0' ↔ '8', '6' ↔ '9', '3' ↔ '5'

**Interpretation:**
- **Success:** Patches from same digit cluster → shared encoder induces semantic clustering
- **Failure (mixed clusters):** Shared encoder not learning structure; increase model capacity
- **Partial success (loose clusters):** Clustering exists but noisy; add regularization
- **Failure (random):** Training issue; check loss curves

**Output file:** `patch_latent_tsne.png`

### 4. Digit-Class Similarity Matrix

**What:** Compute mean embedding per digit class; compute pairwise cosine similarities

**Method:**
1. For each digit class (0-9), average embeddings of all its patches
2. Normalize to unit norm
3. Compute cosine similarity matrix [10×10]
4. Print with row/column labels (digits)

**Expected:**
```
     0     1     2     3     4     5     6     7     8     9
0: [ 1.00  0.15  0.25  0.20  0.10  0.18  0.65  0.12  0.72  0.60 ]
1: [ 0.15  1.00  0.18  0.10  0.30  0.08  0.12  0.45  0.08  0.20 ]
...
8: [ 0.72  0.08  0.30  0.20  0.10  0.15  0.60  0.10  1.00  0.55 ]
9: [ 0.60  0.20  0.25  0.18  0.08  0.12  0.50  0.20  0.55  1.00 ]
```

**Interpretation:**
- Diagonal = 1.0 (each digit similar to itself) ✓
- High similarity within visual groups:
  - (0, 6, 8, 9) ~ 0.55-0.75 ✓ (all have loops/curves)
  - (3, 5) ~ 0.40-0.60 ✓ (both have horizontal strokes)
  - (1, 7) ~ 0.35-0.50 ✓ (both have vertical stroke)
- Low similarity across dissimilar digits:
  - (1, 0) ~ 0.10-0.20 ✓ (straight vs. curved)
  - (4, 0) ~ 0.10 ✓ (angular vs. curved)

---

## Expected Outcomes

### Scenario A: Full Success ✓✓✓

| Metric | Outcome |
|--------|---------|
| Test Accuracy | >95% |
| t-SNE Clustering | 10 well-separated clusters |
| Similarity Matrix | Semantic grouping (0-6-8-9 together, etc.) |
| Attention Patterns | Digit-specific, focused on key structures |

**Conclusion:** Shared encoding + selective attention successfully learns semantic structure

**Next step:** Test on more complex datasets (CIFAR-10, augmented MNIST)

### Scenario B: Partial Success ✓✓

| Metric | Outcome |
|--------|---------|
| Test Accuracy | 90-95% |
| t-SNE Clustering | Loose clusters, some overlap |
| Similarity Matrix | Weak semantic grouping |
| Attention Patterns | Reasonable but not focused |

**Conclusion:** Mechanism is working but model capacity or architecture needs tuning

**Next step:** Increase model size, add regularization, tune glimpse count

### Scenario C: Limited Success ✓

| Metric | Outcome |
|--------|---------|
| Test Accuracy | 80-90% |
| t-SNE Clustering | No clear clustering |
| Similarity Matrix | Random or uniform |
| Attention Patterns | Uniform attention, no structure |

**Conclusion:** Architecture needs fundamental revision

**Diagnostics:**
- Are embeddings learning at all? (check embedding variance)
- Is attention learning? (check attention entropy)
- Is GRU state accumulating? (check state variance across glimpses)

**Next step:** Ablation study; simplify to single-glimpse attention first

### Scenario D: Failure ✗

| Metric | Outcome |
|--------|---------|
| Test Accuracy | <80% |
| Loss | Not decreasing |
| Attention Patterns | Degenerate (uniform or one-hot on random patch) |

**Conclusion:** Training issue or fundamental architecture mismatch

**Diagnostics:**
- Check gradient flow: are attention/encoder weights updating?
- Check learning rate: is it too high/low?
- Check initialization: are embeddings starting random?

**Next step:** Debug training loop; start with simpler baseline

---

## Running the Experiment

### Installation

```bash
python -m venv .venv
source .venv/bin/activate  # or: .venv\Scripts\activate (Windows)

pip install -U pip
pip install "keras>=3" tensorflow numpy matplotlib scikit-learn
```

### Execution

```bash
cd experiments/01_semantic_clustering/
python mnist_selective_feature_poc.py
```

**Expected runtime:**
- Training: ~2-3 minutes (GPU) / ~10-15 minutes (CPU)
- t-SNE: ~1-2 minutes
- Total: ~5 minutes (GPU)

### Output Files

| File | Purpose |
|------|---------|
| `mnist_selective_feature_poc.weights.h5` | Trained model weights (reuse for later experiments) |
| `attention_glimpses.png` | Attention heatmaps for 8 random test images |
| `patch_latent_tsne.png` | t-SNE visualization of 216,000 patches |
| Console output | Classification accuracy + similarity matrix |

---

## Interpretation Guide

### Reading the t-SNE Plot

1. **Ideal (✓✓✓):** 10 distinct clouds, minimal overlap
   - Dense cluster per digit
   - Clear separation between most digit pairs
   - Some expected overlap (0-8, 3-5, 6-9)

2. **Good (✓✓):** Loose clusters with some mixing
   - Clear tendency to group by digit
   - Some cross-digit points (acceptable)
   - Overall structure visible

3. **Weak (✓):** Significant mixing, loose structure
   - Points don't clearly separate by digit
   - But some structure is visible (not random)
   - Model learned something but not strongly

4. **Failure (✗):** Random scatter
   - No visible structure
   - Points mixed uniformly
   - Model didn't learn embeddings

### Reading the Attention Visualization

1. **Well-trained model:**
   - Glimpse 1: Scattered attention across multiple patches
   - Glimpse 2-3: Convergence on specific regions
   - Different digits attend to different regions

2. **Undertrained model:**
   - Attention remains scattered across all 3 glimpses
   - No convergence or pattern

3. **Overfitting (attention too sharp):**
   - Glimpse 1 already highly concentrated on one patch
   - No benefit from additional glimpses

### Reading the Similarity Matrix

**High similarity (0.5-1.0) should occur between:**
- **Loops/Curves:** 0, 6, 8, 9
- **Horizontal strokes:** 3, 5
- **Vertical strokes:** 1, 7
- **Diagonal/Angular:** 2, 4

**Low similarity (0.0-0.3) should occur between:**
- Vertical vs. curved: 1 vs. 0
- Straight vs. looped: 4 vs. 8
- Simple vs. complex: 1 vs. 0

If this pattern is **reversed**, something is wrong.

---

## Ablation Studies (Future)

To understand which components matter, try:

1. **Remove shared encoding:** Use separate Conv layers per patch
   - Expected: Lower clustering (baseline)

2. **Remove selective attention:** Use global average pooling instead
   - Expected: Worse accuracy, uniform attention

3. **Single glimpse:** Set glimpses=1
   - Expected: Worse accuracy, faster training

4. **Larger latent dim:** Increase LATENT_DIM to 128, 256
   - Expected: Tighter clusters, higher accuracy

5. **Larger model:** More filters in Conv2D
   - Expected: Better clustering, slower training

---

## Theoretical Connection

This experiment embodies several key ideas from the semantic context architecture:

| Concept | Implementation |
|---------|-----------------|
| **Shared latent space** | All patches → same Conv2D weights |
| **Semantic context extraction** | Encoder learns to map visual patterns |
| **Hash/addressing** | Implicit: patches with similar content → similar embeddings |
| **Multi-context per input** | Single image has 36 patches, each is a context |
| **Recursive refinement** | Glimpse iterations: $Z_0 \to Z_1 \to Z_2 \to Z_3$ |
| **Autonomous halting** | Model learns to focus (soft attention converges) |
| **Unsupervised structure** | No explicit labels for visual features |

---

## Success Criteria (Checklist)

- [ ] Training completes without NaNs or crashes
- [ ] Test accuracy reaches ≥90%
- [ ] t-SNE shows visible clustering by digit
- [ ] Attention patterns differ between digits
- [ ] Similarity matrix shows semantic grouping
- [ ] Can interpret which patches are important for each digit

---

## Next Steps

### Immediate (If successful)
1. Save model and embeddings for downstream analysis
2. Qualitative inspection: which patches are most informative for each digit?
3. Failure case analysis: misclassified examples

### Short-term (If very successful)
1. Repeat on CIFAR-10 with more complex structure
2. Investigate what patch embeddings have learned (nearest neighbors, ablation)
3. Compare to baseline (separate encoders, no attention)

### Long-term
1. Test on larger datasets (ImageNet subset)
2. Combine with recursive latent refinement ($Z_0 \to Z_1 \to Z_2$)
3. Integrate memory hierarchy (cache frequently-used embeddings)

---

## References & Inspiration

- **Vision Transformers:** Dosovitskiy et al., 2020 (patch-based processing)
- **Recurrent Attention:** Mnih et al., 2014 (selective attention networks)
- **Semantic Hashing:** Salakhutdinov & Hinton, 2009 (hash-based retrieval)
- **Transformer Attention:** Vaswani et al., 2017 (scaled dot-product attention)

---

**Author:** Semantic Context AI Architecture Research  
**Date:** 2026-10-07  
**Status:** Initial PoC
