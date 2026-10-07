# Experiment 01: Adaptive Foveated Attention with Uncertainty-Guided Search

## Overview

This experiment tests the central hypothesis of the semantic context architecture in an image setting:

> Can a model process a single unified foveated image, extract spatially meaningful features from the full canvas, and adaptively move its focal point toward the highest-information region when confidence is insufficient?

This design directly addresses a key issue in naive foveation: if the focus moves, the image changes, so we must not rely on a static embedding of a single focal patch alone. Instead, the model keeps a recurrent state, extracts a full feature map from the entire canvas, and uses that map to decide where to look next.

---

## Core Idea

Instead of treating focus and context as separate input streams, we generate a single 28×28 image that contains:

- a clear focal region around the current attention center
- a blurred but intact peripheral context around it

The result is a single, unified image tensor that preserves spatial layout while emphasizing the currently relevant locus.

This is closer to human vision than a patch list:
- the visual field is not split into separate channels
- the retina does not store "focus data" and "context data" as unrelated streams
- rather, the brain processes a single coherent scene with local detail and global context simultaneously

---

## Why This Matters

A naive approach that simply crops a patch and embeds it is insufficient, because:

- when the focus moves to another region, the input changes
- the same underlying object may produce a different embedding depending on where the model is looking
- that breaks the assumption that semantically related features should remain consistent across glimpses

The solution is not to force the whole concept into one static embedding. Instead:

1. the model processes the full foveated canvas
2. it extracts spatial features from the whole image
3. it identifies the region with the highest information content
4. it shifts the focal point there if the current confidence is low
5. it remembers previous information via recurrent state

This makes focal movement a decision process, not a random or static sampling artifact.

---

## Hypothesis

### H1: Full-canvas spatial features preserve semantic consistency
If we encode the entire fused foveated image rather than just the focal crop, then different glimpses are still grounded in the same global visual field, allowing the recurrent state to integrate context over time.

### H2: Information-driven focus traversal is useful
The model can estimate where the highest-information region is by analyzing the feature map and selecting the next focal location there.

### H3: Recurrent state stabilizes the semantic trajectory
Even when the focal region changes, the model's recurrent state retains previously seen features, allowing the system to accumulate evidence rather than re-starting from scratch on each glance.

---

## Experimental Design

### Architecture Overview

```text
Input Image (28x28)
    ↓
Foveated Image Generator
    - focus region: sharp / original pixels
    - periphery: blurred background
    - output: single unified foveated 28x28 image
    ↓
Unified Foveated Encoder
    - shared Conv2D over full canvas
    - preserves spatial structure
    - outputs feature map + information map
    ↓
Recurrent State (GRU)
    - carries information across glimpses
    ↓
Class Head + Focal-Point Predictor
    - confidence estimate for current digit hypothesis
    - if confidence < threshold, move focus to high-information location
    ↓
Repeat for K glimpses
```

### Key Components

#### 1. Foveated image generator
The model produces a single image where:

- a local region around the current focal point is kept sharp
- the rest of the image is downsampled/blurred and blended into the background

This creates a unified field of view rather than a split representation.

#### 2. Unified full-canvas encoder
The entire 28x28 foveated image is processed by a shared Conv2D stack. This is essential because:

- the model sees the full scene
- spatial coordinates remain intact
- local detail and peripheral context are processed together

This differs from a patch-only approach, which loses global context.

#### 3. Information map
From the feature map, the model estimates a spatial saliency map:

- where are the contours, edges, loops, or stroke intersections?
- which location has the highest useful structure for the current classification problem?

This map drives the next focal movement.

#### 4. Focal point selection
When the model is uncertain, it uses the information map to select the next focus region. This is the core adaptive behavior:

- if prediction is already confident, stop or keep focus stable
- if prediction is uncertain, move to more informative regions

#### 5. Recurrent state accumulation
The model remembers what it has seen before. This stabilizes the interpretation across multiple glimpses and prevents each glance from being treated as a separate, disconnected observation.

---

## Why This Design Is Different

### Not: separate patch streams
This experiment does not create a list of patch embeddings and then treat them as separate objects.

Instead:
- there is one canvas
- one unified image tensor
- one spatial feature map
- one recurrent reasoning process

### Yes: active perception
The model actively decides where to look next. The focus is guided by feature-level information, not by a fixed hand-coded schedule.

This is a more natural analogue of visual reasoning:
- human perception is not a static feedforward pass
- it is a sequence of refined observations
- each new glance updates the internal state and changes the next fixation point

---

## Experimental Setup

### Hyperparameters

| Parameter | Value | Purpose |
|-----------|-------|---------|
| Input size | 28×28 | MNIST canonical size |
| Focal region | 7×7 | sharpened local region |
| Blur level | 7 | low-pass background |
| Glimpses | 3 | sequential visual refinement |
| Latent dim | 64 | embedding capacity |
| Hidden dim | 64 | recurrent state dimension |
| Confidence threshold | 0.90 | move focus when uncertain |
| Batch size | 128 | standard training |
| Epochs | 8 | quick verification |

### Data

- MNIST train/test split
- Inputs normalized to [0,1]
- Training objective: standard digit classification loss

---

## Measurements

### 1. Classification accuracy
Final test accuracy on MNIST remains the primary metric.

### 2. Focal trajectory visualization
Plot the focal point trajectory over successive glimpses. This reveals whether the model learns to move toward discriminative regions.

### 3. Information map evolution
Visualize how the saliency map changes across glimpses. This reveals the model's concept of where structure exists.

### 4. Feature clustering
Use the final feature map to visualize whether digit-like structures cluster in the latent space.

---

## Success Criteria

The experiment is considered successful if:

- test accuracy is strong
- focal points move toward informative regions rather than random locations
- feature maps become structurally organized by digit class
- recurrent state stabilizes the interpretation over glimpses

---

## Expected Outcomes

### Scenario A: Strong success
- accuracy >95%
- first glance near center, later glimpses move toward relevant stroke regions
- saliency map aligns with discriminative parts of the digit
- feature maps form digit-like clusters

### Scenario B: Moderate success
- accuracy still good but trajectory is noisy
- model attends to informative contours but not always consistently

### Scenario C: Weak success
- accuracy acceptable but focal points do not clearly adapt
- feature map lacks structured clustering

### Scenario D: Failure
- model does not learn meaningful trajectory
- saliency map remains diffuse or random
- recurrent state doesn't improve prediction substantially

---

## Interpretation Guide

### Focal-point behavior
Good behavior:
- early glimpses do coarse survey
- later glimpses move toward decisive strokes or loops
- focus does not remain fixed at the center if the model is uncertain

Bad behavior:
- always stays centered
- jumps randomly without structure
- ignores the information map

### Feature map behavior
Good behavior:
- edges, loops, and strokes are visible in the feature map
- similar digits produce similar structure in related regions

Bad behavior:
- feature map is dominated by background blur
- no spatial organization emerges

---

## Theoretical Link to Semantic Context Architecture

This experiment maps directly to the architecture concept:

| Architecture idea | Corresponding mechanism |
|------------------|--------------------------|
| Unified semantic field | Single foveated image tensor |
| Shared latent encoder | Conv2D over full canvas |
| Information extraction | Feature map + saliency map |
| Recursive refinement | Recurrent state across glimpses |
| Decision to move focus | Information-guided focal selection |
| Memory | GRU state |
| Focused perception | Local high-detail region |
| Global context | Blurred periphery retained in same canvas |

The crucial difference from the earlier patch-only design is that the whole image remains encoded as one coherent scene, and the model decides where to inspect next using the full spatial signal, not just isolated local crops.

---

## Next Steps

If this experiment works, the next extensions are:

1. replace the simple foveated generator with a more realistic Gaussian/retina-like weighting
2. add a proper uncertainty head to decide when to halt
3. compare against static patch-only baselines
4. extend to CIFAR-10 or a more complex visual dataset
5. test how recurrent trajectories differ across classes

---

## Summary

This experiment directly tests a more biologically plausible and architecture-consistent model:

- a single unified scene is processed
- high-detail focus and low-detail context are blended in one image tensor
- the model identifies where informative structure lies
- the next fixation location is chosen adaptively
- previous observations are retained through recurrent state

This is a better match to the semantic context hypothesis than a simple patch list because it preserves the full spatial field while still allowing active, uncertainty-driven attention.
