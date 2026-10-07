#!/usr/bin/env python3
"""
MNIST Selective Feature PoC (Proof of Concept)

This experiment tests the core hypothesis of the semantic context architecture:
  - Can we use ONE shared Conv2D encoder on multiple image patches?
  - Do feature embeddings naturally cluster by digit without supervision?
  - Can selective attention (glimpses) learn meaningful sampling patterns?

Architecture flow:
  image
    -> candidate patches (via extract_patches)
    -> ONE shared Conv2D applied to each patch
    -> patch latent vectors (shared embedding space)
    -> learned selective attention (GLIMPSES iterations)
    -> recurrent state (GRU)
    -> digit classifier

Key insight:
  By forcing all patches through identical weights, we create a shared latent space.
  Related visual features (e.g., curves in '0' vs curves in '8') should naturally
  cluster together in this space without explicit supervision.

After training:
  1) Test accuracy on MNIST
  2) Attention visualisation: which patches does the model attend to?
  3) t-SNE of patch embeddings, coloured by digit class
  4) Similarity matrix: do digit classes form coherent regions?

Installation:
  python -m venv .venv_mnist_poc
  source .venv_mnist_poc/bin/activate
  pip install -U pip
  pip install "keras>=3" tensorflow numpy matplotlib scikit-learn

Usage:
  python mnist_selective_feature_poc.py

Optional:
  KERAS_BACKEND=tensorflow python mnist_selective_feature_poc.py
"""

import os
os.environ.setdefault("KERAS_BACKEND", "tensorflow")

import numpy as np
import matplotlib.pyplot as plt
import keras
from keras import layers, ops
from sklearn.manifold import TSNE


# ========================================
# Configuration
# ========================================
SEED = 42
IMG_SIZE = 28
PATCH = 7
STRIDE = 4

LATENT_DIM = 64
HIDDEN_DIM = 64
GLIMPSES = 3

BATCH_SIZE = 128
EPOCHS = 8

TSNE_SAMPLES = 6000

keras.utils.set_random_seed(SEED)


# ========================================
# 1. Candidate patch extraction
# ========================================
class PatchExtractor(layers.Layer):
    """
    Extract overlapping patches from the input image using sliding window.

    Args:
      patch: size of each patch (patch x patch)
      stride: step size for the sliding window

    Input shape: [B, 28, 28, 1]
    Output shape: [B, num_patches, patch, patch, 1]
    
    Example:
      For 28x28 image, patch=7, stride=4:
      - num_patches = ((28 - 7) // 4 + 1)^2 = 6^2 = 36
    """
    def __init__(self, patch=7, stride=4, **kwargs):
        super().__init__(**kwargs)
        self.patch = patch
        self.stride = stride

    def call(self, x):
        # x: [B, 28, 28, 1]
        p = ops.image.extract_patches(
            x,
            size=(self.patch, self.patch),
            strides=(self.stride, self.stride),
            padding="valid",
        )
        # extract_patches returns: [B, rows, cols, patch*patch]
        b = ops.shape(p)[0]
        p = ops.reshape(p, (b, -1, self.patch, self.patch, 1))
        # [B, num_patches, patch, patch, 1]
        return p


# ========================================
# 2. Shared patch encoder
# ========================================
class SharedPatchEncoder(layers.Layer):
    """
    The core component: ONE shared Conv2D layer applied to ALL patches.

    This constraint is critical: every candidate patch uses identical weights.
    This creates the "shared latent semantic space" where:
      - Similar visual patterns (curves, edges, corners) map to similar embeddings
      - Unsupervised clustering of semantic concepts emerges

    Args:
      latent_dim: dimension of output embedding for each patch

    Input shape: [B, num_patches, patch_h, patch_w, channels]
    Output shape: [B, num_patches, latent_dim]

    Internals:
      - Conv2D(16 filters, 3x3): extract local features from each patch
      - GlobalAveragePooling2D: aggregate spatial dimensions
      - Dense(latent_dim): project to latent space
    """

    def __init__(self, latent_dim=64, **kwargs):
        super().__init__(**kwargs)

        self.conv = layers.Conv2D(
            filters=16,
            kernel_size=3,
            padding="same",
            activation="relu",
        )

        self.pool = layers.GlobalAveragePooling2D()
        self.proj = layers.Dense(latent_dim)

    def call(self, patches):
        # Input: [B, N, P, P, 1]
        # N = number of patches, P = patch size
        b = ops.shape(patches)[0]
        n = ops.shape(patches)[1]

        # Flatten batch and patches for shared processing
        x = ops.reshape(patches, (-1, PATCH, PATCH, 1))
        x = self.conv(x)                 # ONE Conv layer (shared weights)
        x = self.pool(x)
        z = self.proj(x)

        # Reshape back to [B, N, latent_dim]
        z = ops.reshape(z, (b, n, -1))
        return z


# ========================================
# 3. Recurrent selective attention
# ========================================
class SelectiveGlimpse(layers.Layer):
    """
    Recurrent attention mechanism: at each step (glimpse), decide which patch to attend.

    At each step:
      1. Use current recurrent state as a query
      2. Score every patch (key) against this query
      3. Softmax attention: which patches are relevant?
      4. Weighted combination: aggregate selected features
      5. GRU update: integrate the glimpse into new recurrent state

    Args:
      latent_dim: dimension of patch embeddings
      hidden_dim: dimension of recurrent state

    During training: soft attention (weighted sum over all patches)
    During inference: argmax attention (pick single patch), but we use soft for gradients

    The model learns:
      - How to query: query() transforms state to question
      - How to match: key() transforms features to comparable representation
      - How to select: score() assigns relevance to each patch
    """

    def __init__(self, latent_dim=64, hidden_dim=64, **kwargs):
        super().__init__(**kwargs)

        self.query = layers.Dense(hidden_dim, use_bias=False)
        self.key = layers.Dense(hidden_dim, use_bias=False)
        self.score = layers.Dense(1, use_bias=False)

        self.gru = layers.GRUCell(hidden_dim)

        self.feature_to_hidden = layers.Dense(hidden_dim)
        self.norm = layers.LayerNormalization()

    def call(self, z, state):
        # z: [B, N, D] patch embeddings
        # state: [B, H] current recurrent state

        q = self.query(state)[:, None, :]           # [B, 1, H]
        k = self.key(z)                             # [B, N, H]

        h = ops.tanh(k + q)                          # broadcast addition
        logits = self.score(h)
        logits = ops.squeeze(logits, axis=-1)       # [B, N]

        # Soft attention (differentiable, used during training)
        attention = ops.softmax(logits, axis=-1)    # [B, N]

        # Weighted glimpse: sum of attended patches
        glimpse = ops.sum(z * attention[..., None], axis=1)  # [B, D]
        glimpse_h = self.feature_to_hidden(glimpse)

        # GRU: integrate this glimpse
        _, new_state = self.gru(glimpse_h, [state])
        new_state = self.norm(new_state[0])

        return new_state, attention, glimpse


# ========================================
# 4. Complete model
# ========================================
class SelectiveMNIST(keras.Model):
    """
    End-to-end model for MNIST digit classification using selective attention.

    Flow:
      1. Extract patches from 28x28 image
      2. Encode each patch with shared Conv2D -> latent space
      3. Apply GLIMPSES selective attention iterations
         - Each iteration: query state, attend to patches, update state
      4. Final state fed to classifier

    This is a proof of concept for the semantic context architecture:
      - Shared encoding ensures all patches map to common latent space
      - Attention learns which patches are diagnostic
      - Recurrent state accumulates information across glimpses
    """
    def __init__(
        self,
        patch=7,
        stride=4,
        latent_dim=64,
        hidden_dim=64,
        glimpses=3,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.extractor = PatchExtractor(patch, stride)
        self.encoder = SharedPatchEncoder(latent_dim)

        self.glimpse = SelectiveGlimpse(
            latent_dim=latent_dim,
            hidden_dim=hidden_dim,
        )

        self.initial_state = layers.Dense(hidden_dim, activation="tanh")
        self.classifier = layers.Dense(10)

        self.glimpses = glimpses
        self.hidden_dim = hidden_dim

    def call(self, x, training=False, return_attention=False):
        """
        Args:
          x: [B, 28, 28, 1] input images
          training: whether in training mode
          return_attention: if True, return attention maps and embeddings for visualization

        Returns:
          logits: [B, 10] class logits
          (optional) attention, glimpses, patch_embeddings, patches
        """
        # 1. Extract patches
        patches = self.extractor(x)  # [B, N, 7, 7, 1]

        # 2. Shared encoding: all patches -> latent space
        z = self.encoder(patches)    # [B, N, latent_dim]

        # 3. Initialize recurrent state from global mean
        state = self.initial_state(ops.mean(z, axis=1))  # [B, hidden_dim]

        all_attention = []
        all_glimpses = []

        # 4. Multiple selective attention iterations
        for _ in range(self.glimpses):
            state, attention, glimpse = self.glimpse(z, state)
            all_attention.append(attention)
            all_glimpses.append(glimpse)

        # 5. Classify from final recurrent state
        logits = self.classifier(state)

        if return_attention:
            att = ops.stack(all_attention, axis=1)   # [B, glimpses, num_patches]
            gl = ops.stack(all_glimpses, axis=1)     # [B, glimpses, latent_dim]
            return logits, att, gl, z, patches

        return logits


# ========================================
# 5. Training
# ========================================
def train():
    """Train the model on MNIST."""
    print("Loading MNIST...")
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

    x_train = x_train.astype("float32") / 255.0
    x_test = x_test.astype("float32") / 255.0

    x_train = x_train[..., None]
    x_test = x_test[..., None]

    model = SelectiveMNIST(
        patch=PATCH,
        stride=STRIDE,
        latent_dim=LATENT_DIM,
        hidden_dim=HIDDEN_DIM,
        glimpses=GLIMPSES,
    )

    model.compile(
        optimizer=keras.optimizers.Adam(1e-3),
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=["accuracy"],
    )

    model.build((None, 28, 28, 1))
    model.summary()

    model.fit(
        x_train,
        y_train,
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        validation_split=0.1,
        verbose=2,
    )

    loss, acc = model.evaluate(x_test, y_test, verbose=0)
    print(f"\nTest accuracy: {acc:.4f}")

    return model, x_test, y_test


# ========================================
# 6. Attention visualisation
# ========================================
def show_attention(model, x, y, count=8):
    """
    Visualize which patches the model attends to at each glimpse.

    For each test image:
      - Left: original image with predicted/true label
      - Columns 2-4: attention heatmaps for glimpses 1-3
        * Each patch center is shown as a scatter point
        * Size/alpha of scatter point indicates attention weight
        * Rectangle shows the single most-attended patch (argmax)

    This shows that the model learns meaningful attention patterns:
      - Early glimpses may attend to corners/edges
      - Later glimpses may attend to central features
      - Patterns differ by digit class
    """
    n = min(count, len(x))

    logits, attention, _, _, patches = model(
        x[:n],
        return_attention=True,
        training=False,
    )

    pred = np.argmax(np.asarray(logits), axis=-1)
    attention = np.asarray(attention)

    rows = n
    fig, axes = plt.subplots(rows, GLIMPSES + 1, figsize=(3 * (GLIMPSES + 1), 2.6 * rows))

    if rows == 1:
        axes = np.expand_dims(axes, 0)

    # Candidate patch grid dimensions
    patch_grid = (IMG_SIZE - PATCH) // STRIDE + 1

    for i in range(rows):
        # First column: original image
        axes[i, 0].imshow(x[i, ..., 0], cmap="gray")
        axes[i, 0].set_title(f"true={y[i]} pred={pred[i]}")
        axes[i, 0].axis("off")

        # Remaining columns: attention for each glimpse
        for g in range(GLIMPSES):
            ax = axes[i, g + 1]

            heat = attention[i, g].reshape(patch_grid, patch_grid)

            ax.imshow(x[i, ..., 0], cmap="gray")

            # Patch centre coordinates
            ys, xs = np.meshgrid(
                np.arange(patch_grid) * STRIDE + PATCH / 2,
                np.arange(patch_grid) * STRIDE + PATCH / 2,
                indexing="ij",
            )

            # Scatter plot: patch centres, sized by attention weight
            ax.scatter(
                xs.ravel(),
                ys.ravel(),
                s=500 * heat.ravel() + 2,
                alpha=0.55,
            )

            # Highlight most-attended patch with rectangle
            selected = np.argmax(attention[i, g])
            sy = selected // patch_grid
            sx = selected % patch_grid

            rect = plt.Rectangle(
                (sx * STRIDE, sy * STRIDE),
                PATCH,
                PATCH,
                fill=False,
                linewidth=2,
            )
            ax.add_patch(rect)

            ax.set_title(f"glimpse {g+1}")
            ax.set_xlim(0, IMG_SIZE)
            ax.set_ylim(IMG_SIZE, 0)
            ax.axis("off")

    plt.tight_layout()
    plt.savefig("attention_glimpses.png", dpi=160)
    plt.show()
    print("\nSaved: attention_glimpses.png")


# ========================================
# 7. t-SNE visualisation of patch embeddings
# ========================================
def plot_tsne(model, x, y, max_samples=6000):
    """
    Visualize learned patch embeddings in 2D using t-SNE.

    Hypothesis:
      Because all patches use the same encoder, patches with similar visual content
      should map to similar regions in latent space, even without explicit supervision.

    Each patch is colored by the digit it came from. If the hypothesis is correct:
      - Patches from '0' should cluster together
      - Patches from '1' should form a separate cluster
      - Shared features (e.g., curves) between '0' and '8' should be nearby

    This is a diagnostic visualization that tests:
      "Do semantically similar features naturally cluster in the shared space?"
    """
    print("\nCollecting patch latent vectors for t-SNE...")

    n = min(max_samples, len(x))

    logits, attention, _, z, _ = model(
        x[:n],
        return_attention=True,
        training=False,
    )

    z = np.asarray(z)                 # [N, patches, latent_dim]
    y0 = np.asarray(y[:n])

    # Flatten patches and their labels
    patches_per_image = z.shape[1]
    z = z.reshape(-1, z.shape[-1])
    labels = np.repeat(y0, patches_per_image)

    # Limit for computational efficiency
    if len(z) > max_samples:
        rng = np.random.default_rng(SEED)
        idx = rng.choice(len(z), max_samples, replace=False)
        z = z[idx]
        labels = labels[idx]

    print(f"Running t-SNE on {len(z):,} patch vectors...")

    tsne = TSNE(
        n_components=2,
        perplexity=30,
        init="pca",
        random_state=SEED,
        learning_rate="auto",
    )

    emb = tsne.fit_transform(z)

    plt.figure(figsize=(10, 8))

    colors = plt.cm.tab10(np.arange(10))

    for digit in range(10):
        m = labels == digit
        plt.scatter(
            emb[m, 0],
            emb[m, 1],
            s=5,
            alpha=0.35,
            label=str(digit),
            color=colors[digit],
        )

    plt.title("t-SNE of learned patch latent features\n(colored by digit class)")
    plt.xlabel("t-SNE component 1")
    plt.ylabel("t-SNE component 2")
    plt.legend(markerscale=3, title="digit", ncol=5)
    plt.tight_layout()
    plt.savefig("patch_latent_tsne.png", dpi=180)
    plt.show()
    print("Saved: patch_latent_tsne.png")


# ========================================
# 8. Quantitative patch statistics
# ========================================
def print_patch_statistics(model, x, y):
    """
    Compute mean latent vector per digit class and their pairwise similarities.

    This provides a simple numerical check:
      - If semantic clustering works, similar digits (0,8,6,9) should have high cosine sim
      - Dissimilar digits (0,1) should have low cosine sim

    Output:
      Cosine similarity matrix [10x10] where entry (i,j) is similarity between
      mean embeddings of digit i and digit j.
    """
    _, _, _, z, _ = model(
        x,
        return_attention=True,
        training=False,
    )

    z = np.asarray(z)
    y = np.asarray(y)

    means = []

    for digit in range(10):
        zd = z[y == digit].reshape(-1, z.shape[-1])
        m = zd.mean(axis=0)
        m /= np.linalg.norm(m) + 1e-8
        means.append(m)

    means = np.asarray(means)
    sim = means @ means.T

    print("\n" + "="*60)
    print("Mean latent cosine similarity by digit class")
    print("(Higher = more similar; similar digits should cluster)")
    print("="*60)
    np.set_printoptions(precision=2, suppress=True)
    print("     ", " ".join(f"{i:6d}" for i in range(10)))
    for i, row in enumerate(sim):
        print(f"{i}: {row}")
    print("="*60)


# ========================================
# Main
# ========================================
if __name__ == "__main__":
    print("\n" + "="*70)
    print("MNIST Selective Feature PoC")
    print("Testing: shared latent space + selective attention")
    print("="*70 + "\n")

    model, x_test, y_test = train()

    # Save trained weights
    model.save_weights("mnist_selective_feature_poc.weights.h5")
    print("\nSaved: mnist_selective_feature_poc.weights.h5")

    print("\n" + "="*70)
    print("Visualizing attention patterns")
    print("="*70)
    show_attention(model, x_test, y_test, count=8)

    print("\n" + "="*70)
    print("Visualizing learned embedding space")
    print("="*70)
    plot_tsne(
        model,
        x_test,
        y_test,
        max_samples=TSNE_SAMPLES,
    )

    print("\n" + "="*70)
    print("Quantitative analysis")
    print("="*70)
    print_patch_statistics(
        model,
        x_test[:2000],
        y_test[:2000],
    )

    print("\n" + "="*70)
    print("Experiment complete!")
    print("="*70)
    print("\nGenerated artifacts:")
    print("  - mnist_selective_feature_poc.weights.h5  (trained weights)")
    print("  - attention_glimpses.png                  (attention visualizations)")
    print("  - patch_latent_tsne.png                   (embedding space)")
    print("\nInterpretation:")
    print("  1. Test accuracy shows if selective attention works")
    print("  2. attention_glimpses.png: do attended patches make sense?")
    print("  3. patch_latent_tsne.png: do similar digits cluster?")
    print("  4. Similarity matrix: quantitative coherence check")
