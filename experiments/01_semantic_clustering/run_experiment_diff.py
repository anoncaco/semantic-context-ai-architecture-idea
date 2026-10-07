"""
Experiment 01-B: Fully Differentiable Adaptive Foveated Attention
- Spatial Softmax를 통한 Expected Coordinates 계산
- 2D Gaussian Mask 기반 Differentiable Foveation Blending
- Classification Loss 기반 End-to-End Attention Optimization
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist, squareform
import warnings
warnings.filterwarnings('ignore')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# ============================================================
# 1. 합성 숫자 데이터 생성
# ============================================================
def generate_digit(digit, noise=0.12):
    img = np.zeros((28, 28), dtype=np.float32)
    if digit == 0:
        yy, xx = np.ogrid[:28, :28]
        dist = np.sqrt((yy - 14)**2 + (xx - 14)**2)
        img[(dist > 6) & (dist < 10)] = 1.0
    elif digit == 1:
        img[5:23, 13:16] = 1.0
    elif digit == 2:
        img[5:8, 7:21] = 1.0
        img[7:14, 18:21] = 1.0
        img[13:16, 7:21] = 1.0
        img[15:22, 7:10] = 1.0
        img[19:22, 7:21] = 1.0
    elif digit == 3:
        img[5:8, 7:21] = 1.0
        img[7:13, 18:21] = 1.0
        img[12:15, 10:21] = 1.0
        img[14:20, 18:21] = 1.0
        img[19:22, 7:21] = 1.0
    elif digit == 4:
        img[5:15, 7:10] = 1.0
        img[12:15, 7:21] = 1.0
        img[5:23, 18:21] = 1.0
    elif digit == 5:
        img[5:8, 7:21] = 1.0
        img[7:13, 7:10] = 1.0
        img[12:15, 7:21] = 1.0
        img[14:20, 18:21] = 1.0
        img[19:22, 7:21] = 1.0
    elif digit == 6:
        img[5:22, 7:10] = 1.0
        img[5:8, 7:21] = 1.0
        img[12:15, 7:21] = 1.0
        img[14:22, 18:21] = 1.0
        img[19:22, 7:21] = 1.0
    elif digit == 7:
        img[5:8, 7:21] = 1.0
        img[5:23, 18:21] = 1.0
    elif digit == 8:
        yy, xx = np.ogrid[:28, :28]
        dist1 = np.sqrt((yy - 9)**2 + (xx - 14)**2)
        dist2 = np.sqrt((yy - 18)**2 + (xx - 14)**2)
        img[((dist1 > 4.5) & (dist1 < 7.5)) | ((dist2 > 4.5) & (dist2 < 7.5))] = 1.0
    elif digit == 9:
        img[5:15, 7:10] = 1.0
        img[5:8, 7:21] = 1.0
        img[12:15, 7:21] = 1.0
        img[5:23, 18:21] = 1.0
        
    img += np.random.randn(28, 28).astype(np.float32) * noise
    return np.clip(img, 0, 1)

def create_dataset(n_per_class=120, noise=0.12, seed=42):
    np.random.seed(seed)
    images, labels = [], []
    for d in range(10):
        for _ in range(n_per_class):
            images.append(generate_digit(d, noise))
            labels.append(d)
    images = np.stack(images)
    labels = np.array(labels)
    idx = np.random.permutation(len(images))
    return images[idx], labels[idx]

train_imgs, train_lbls = create_dataset(n_per_class=120, noise=0.12, seed=42)
test_imgs, test_lbls = create_dataset(n_per_class=40, noise=0.12, seed=123)

print(f"Train: {train_imgs.shape}, Test: {test_imgs.shape}")

# ============================================================
# 2. 완전 미분 가능한 모델 정의
# ============================================================
class DifferentiableFoveatedAttention(nn.Module):
    """
    KEY IMPROVEMENTS from run_experiment.py:
    
    1. Differentiable Foveation Blending
       OLD: Hard overlay with discrete indexing (non-differentiable focal point)
       NEW: 2D Gaussian Mask for soft blending (fully differentiable)
       
    2. Expected Focus Point Calculation
       OLD: Discrete argmax on info map (no gradient flow)
       NEW: Spatial softmax to compute expected coordinates (smooth gradients)
       
    3. Soft Gating for Focus Movement
       OLD: Hard threshold on confidence (discrete decision)
       NEW: Soft gating: focus = gate * next_focus + (1-gate) * focus
       
    4. End-to-End Differentiability
       OLD: Focus movement separated from loss (two-stage optimization)
       NEW: Entire pipeline differentiable, attention optimized by classification loss
    """
    def __init__(self, latent_dim=32, hidden_dim=32, sigma=3.0):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.sigma = sigma
        
        # Grid 좌표 사전 생성 (28x28) - 재사용되므로 buffer로 등록
        y_grid, x_grid = torch.meshgrid(
            torch.arange(28, dtype=torch.float32),
            torch.arange(28, dtype=torch.float32),
            indexing='ij'
        )
        self.register_buffer('y_grid', y_grid)
        self.register_buffer('x_grid', x_grid)

        # Full-canvas encoder (동일)
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 14x14
            nn.Conv2d(32, latent_dim, 3, padding=1),
            nn.ReLU(),
        )

        # Information / Saliency Map Head (동일)
        self.info_head = nn.Conv2d(latent_dim, 1, 1)
        self.proj = nn.Linear(latent_dim, hidden_dim)
        self.gru = nn.GRUCell(hidden_dim, hidden_dim)
        self.classifier = nn.Linear(hidden_dim, 10)

    def create_foveated_differentiable(self, images, focuses):
        """
        IMPROVED: Fully differentiable foveation via 2D Gaussian mask
        
        images: (B, 28, 28)
        focuses: (B, 2) -> (y, x) continuous coordinates (gradient flows through this)
        
        Key change: No discrete indexing. Instead:
        - Create Gaussian mask centered at continuous focus point
        - Soft blend: foveated = mask * original + (1-mask) * blurred
        - Entire operation is differentiable w.r.t. focus coordinates
        """
        B = images.size(0)
        x_in = images.unsqueeze(1)  # (B, 1, 28, 28)
        
        # 1. Background Blur (Average Pooling)
        blurred = F.avg_pool2d(x_in, 5, stride=1, padding=2)
        blurred = F.avg_pool2d(blurred, 3, stride=1, padding=1)

        # 2. 2D Gaussian Mask 생성 (Fully Differentiable)
        fy = focuses[:, 0].view(B, 1, 1)  # (B, 1, 1)
        fx = focuses[:, 1].view(B, 1, 1)  # (B, 1, 1)
        
        # Compute squared distance from each pixel to focus point
        dist_sq = (self.y_grid.unsqueeze(0) - fy)**2 + (self.x_grid.unsqueeze(0) - fx)**2
        
        # Gaussian mask: exp(-dist^2 / 2*sigma^2)
        # This creates smooth gradient field from focus point
        mask = torch.exp(-dist_sq / (2 * (self.sigma ** 2))).unsqueeze(1)  # (B, 1, 28, 28)

        # 3. Soft-blending (Differentiable)
        # High mask value (near focus) -> original
        # Low mask value (far from focus) -> blurred
        foveated = mask * x_in + (1.0 - mask) * blurred
        return foveated

    def spatial_softmax(self, info_map):
        """
        IMPROVED: Compute expected focus point via spatial softmax
        
        info_map: (B, 14, 14) saliency/information map from encoder
        Output: (B, 2) expected (y, x) coordinates in [0, 27] range
        
        Why this is better:
        - OLD method: argmax(info_map) -> discrete point, no gradient
        - NEW method: E[position] via softmax -> smooth, differentiable
        - Gradients flow: loss -> classifier -> info_map -> next_focus -> foveation
        """
        B, H, W = info_map.shape
        
        # Normalize to probability distribution
        probs = F.softmax(info_map.view(B, -1), dim=-1).view(B, H, W)
        
        # Create coordinate grids for 14x14 feature map
        # Map back to 28x28 image space (upsampled)
        y_coords = torch.linspace(0, 27, H, device=info_map.device)
        x_coords = torch.linspace(0, 27, W, device=info_map.device)
        
        y_grid, x_grid = torch.meshgrid(y_coords, x_coords, indexing='ij')
        
        # Compute expected position: E[y] = sum(y * p(y)), E[x] = sum(x * p(x))
        exp_y = torch.sum(probs * y_grid, dim=[1, 2])
        exp_x = torch.sum(probs * x_grid, dim=[1, 2])
        
        return torch.stack([exp_y, exp_x], dim=-1)

    def forward(self, images, num_glimpses=3, threshold=0.70, return_all=False):
        B = images.size(0)
        device = images.device

        # Start at center
        focus = torch.full((B, 2), 14.0, device=device, requires_grad=True)
        h = torch.zeros(B, self.hidden_dim, device=device)

        all_logits, all_focuses, all_embs = [], [focus.clone().detach()], []

        for g in range(num_glimpses):
            # Create foveated image (differentiable)
            foveated = self.create_foveated_differentiable(images, focus)
            
            # Full-canvas encoding
            feat_map = self.encoder(foveated)  # (B, C, 14, 14)
            info_map = self.info_head(feat_map).squeeze(1)  # (B, 14, 14)

            # Accumulate state
            emb = self.proj(feat_map.mean(dim=[2, 3]))
            h = self.gru(emb, h)
            logits = self.classifier(h)
            confidence = F.softmax(logits, dim=-1).max(dim=-1).values

            all_logits.append(logits)
            all_embs.append(h)

            # Adaptive focus movement
            if g < num_glimpses - 1:
                # Expected next focus via spatial softmax
                next_focus = self.spatial_softmax(info_map)
                
                # Soft gating: move focus only when uncertain
                # This creates smooth transition, not hard switch
                gate = (confidence < threshold).float().unsqueeze(1)
                focus = gate * next_focus + (1.0 - gate) * focus
                
                all_focuses.append(focus.clone().detach())

        if return_all:
            return all_logits, all_focuses, all_embs
        return all_logits[-1]


# ============================================================
# 3. 학습
# ============================================================
model = DifferentiableFoveatedAttention(latent_dim=32, hidden_dim=32, sigma=3.0).to(device)
optimizer = optim.Adam(model.parameters(), lr=0.002)
criterion = nn.CrossEntropyLoss()

BATCH_SIZE = 64
EPOCHS = 15
N = len(train_imgs)

print("\n===== Training (Fully Differentiable) =====")
for epoch in range(EPOCHS):
    model.train()
    idx = np.random.permutation(N)
    total_loss, correct, total = 0.0, 0, 0
    
    for i in range(0, N, BATCH_SIZE):
        batch_imgs = torch.from_numpy(train_imgs[idx[i:i+BATCH_SIZE]]).float().to(device)
        batch_lbls = torch.from_numpy(train_lbls[idx[i:i+BATCH_SIZE]]).long().to(device)
        
        optimizer.zero_grad()
        all_logits, _, _ = model(batch_imgs, num_glimpses=3, threshold=0.70, return_all=True)
        
        # Multi-glimpse loss (all intermediate predictions contribute)
        loss = sum(criterion(logits, batch_lbls) for logits in all_logits) / len(all_logits)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        total_loss += loss.item() * len(batch_imgs)
        preds = all_logits[-1].argmax(dim=1)
        correct += (preds == batch_lbls).sum().item()
        total += len(batch_imgs)
    
    print(f"Epoch {epoch+1:2d} | Loss: {total_loss/total:.4f} | Acc: {100*correct/total:.1f}%")


# ============================================================
# 4. 테스트 및 피처 추출
# ============================================================
print("\n===== Evaluation =====")
model.eval()
with torch.no_grad():
    test_tensor = torch.from_numpy(test_imgs).float().to(device)
    all_logits, all_focuses, all_embs = model(
        test_tensor, num_glimpses=3, threshold=0.70, return_all=True
    )
    
    preds = all_logits[-1].argmax(dim=1).cpu().numpy()
    test_acc = (preds == test_lbls).mean()
    print(f"Test Accuracy: {test_acc*100:.1f}%")
    
    final_embs = all_embs[-1].cpu().numpy()


# ============================================================
# 5. 피처 클러스터링 품질 측정
# ============================================================
def clustering_quality(embs, labels):
    dists = squareform(pdist(embs, metric='euclidean'))
    intra_sum, intra_cnt = 0.0, 0
    inter_sum, inter_cnt = 0.0, 0
    n = len(labels)
    for i in range(n):
        for j in range(i+1, n):
            d = dists[i, j]
            if labels[i] == labels[j]:
                intra_sum += d
                intra_cnt += 1
            else:
                inter_sum += d
                inter_cnt += 1
    return intra_sum / max(intra_cnt, 1), inter_sum / max(inter_cnt, 1)

intra, inter = clustering_quality(final_embs, test_lbls)
print(f"\nMean Intra-class distance : {intra:.3f}")
print(f"Mean Inter-class distance : {inter:.3f}")
print(f"Separation ratio (inter/intra) : {inter/intra:.2f}x")


# ============================================================
# 6. PCA + 클러스터링 시각화
# ============================================================
def simple_pca(X, n_components=2):
    X_centered = X - X.mean(axis=0)
    cov = np.cov(X_centered, rowvar=False)
    eigvals, eigvecs = np.linalg.eigh(cov)
    idx = np.argsort(eigvals)[::-1]
    return X_centered @ eigvecs[:, idx[:n_components]]

emb_2d = simple_pca(final_embs, 2)

plt.figure(figsize=(10, 8))
colors = plt.cm.tab10(np.linspace(0, 1, 10))
for d in range(10):
    mask = test_lbls == d
    plt.scatter(emb_2d[mask, 0], emb_2d[mask, 1],
                c=[colors[d]], label=str(d), alpha=0.75, s=35)
plt.legend(title="Digit", loc='best', fontsize=9)
plt.title(f"Feature Clustering (Differentiable Attention)\nTest Acc: {test_acc*100:.1f}%")
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('experiments/01_semantic_clustering/feature_clustering_diff.png', dpi=130)
print("Saved: feature_clustering_diff.png")
plt.close()


# ============================================================
# 7. 초점 궤적 시각화
# ============================================================
fig, axes = plt.subplots(2, 5, figsize=(13, 5.5))
for i, d in enumerate(range(10)):
    ax = axes[i // 5, i % 5]
    idx = np.where(test_lbls == d)[0][0]
    ax.imshow(test_imgs[idx], cmap='gray')
    
    for g, foc in enumerate(all_focuses):
        fy, fx = foc[idx].cpu().numpy()
        color = 'red' if g == 0 else 'yellow'
        ax.plot(fx, fy, 'o', color=color, markersize=9, markeredgecolor='black')
        if g > 0:
            prev = all_focuses[g-1][idx].cpu().numpy()
            ax.plot([prev[1], fx], [prev[0], fy], 'r-', linewidth=1.8)
    
    ax.set_title(f"Digit {d}", fontsize=11)
    ax.axis('off')

plt.suptitle("Focal Point Trajectories (Differentiable: 빨강→노랑→초록)", fontsize=13)
plt.tight_layout()
plt.savefig('experiments/01_semantic_clustering/focal_trajectories_diff.png', dpi=130)
print("Saved: focal_trajectories_diff.png")
plt.close()

print("\n===== 완료 =====")
