"""
Experiment 01: Adaptive Foveated Attention with Uncertainty-Guided Search
합성 MNIST 스타일 데이터로 구현 (실제 MNIST 다운로드가 불가능한 환경 대응)
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
# 1. 합성 숫자 데이터 생성 (MNIST 스타일)
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
# 2. 모델 정의
# ============================================================
class AdaptiveFoveatedAttention(nn.Module):
    def __init__(self, latent_dim=32, hidden_dim=32, focal_size=7):
        super().__init__()
        self.focal_size = focal_size
        self.hidden_dim = hidden_dim
        
        # Full-canvas encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),                    # 14x14
            nn.Conv2d(32, latent_dim, 3, padding=1),
            nn.ReLU(),
        )
        
        # Information / saliency map head
        self.info_head = nn.Conv2d(latent_dim, 1, 1)
        
        # Projection to GRU input
        self.proj = nn.Linear(latent_dim, hidden_dim)
        
        # Recurrent state
        self.gru = nn.GRUCell(hidden_dim, hidden_dim)
        
        # Classification head
        self.classifier = nn.Linear(hidden_dim, 10)
    
    def create_foveated(self, images, focuses):
        """
        images: (B, 28, 28)
        focuses: (B, 2)  y, x
        """
        B = images.size(0)
        half = self.focal_size // 2
        
        # Simple blur (average pooling)
        x = images.unsqueeze(1)
        blurred = F.avg_pool2d(x, 5, stride=1, padding=2)
        blurred = F.avg_pool2d(blurred, 3, stride=1, padding=1)
        
        foveated = blurred.clone()
        
        for b in range(B):
            cy = int(torch.clamp(focuses[b, 0], half, 27 - half).item())
            cx = int(torch.clamp(focuses[b, 1], half, 27 - half).item())
            foveated[b, 0, cy-half:cy+half+1, cx-half:cx+half+1] = \
                images[b, cy-half:cy+half+1, cx-half:cx+half+1]
        
        return foveated
    
    def forward(self, images, num_glimpses=3, threshold=0.70, return_all=False):
        B = images.size(0)
        device = images.device
        
        # 시작 초점은 중앙
        focus = torch.full((B, 2), 14.0, device=device)
        h = torch.zeros(B, self.hidden_dim, device=device)
        
        all_logits = []
        all_focuses = [focus.clone()]
        all_embs = []
        all_info_maps = []
        
        for g in range(num_glimpses):
            foveated = self.create_foveated(images, focus)
            
            feat_map = self.encoder(foveated)               # B, C, 14, 14
            info_map = self.info_head(feat_map).squeeze(1)  # B, 14, 14
            
            emb = self.proj(feat_map.mean(dim=[2, 3]))      # B, hidden
            h = self.gru(emb, h)
            
            logits = self.classifier(h)
            confidence = F.softmax(logits, dim=-1).max(dim=-1).values
            
            all_logits.append(logits)
            all_embs.append(h.clone())
            all_info_maps.append(info_map)
            
            # 다음 초점 결정 (자신감이 낮을 때만 이동)
            if g < num_glimpses - 1:
                info_up = F.interpolate(
                    info_map.unsqueeze(1), size=(28, 28),
                    mode='bilinear', align_corners=False
                ).squeeze(1)
                
                for b in range(B):
                    if confidence[b] < threshold:
                        flat_idx = info_up[b].reshape(-1).argmax()
                        y = (flat_idx // 28).float()
                        x = (flat_idx % 28).float()
                        focus[b, 0] = torch.clamp(y, 4, 23)
                        focus[b, 1] = torch.clamp(x, 4, 23)
                
                all_focuses.append(focus.clone())
        
        if return_all:
            return all_logits, all_focuses, all_embs, all_info_maps
        return all_logits[-1]


# ============================================================
# 3. 학습
# ============================================================
model = AdaptiveFoveatedAttention(latent_dim=32, hidden_dim=32).to(device)
optimizer = optim.Adam(model.parameters(), lr=0.002)
criterion = nn.CrossEntropyLoss()

BATCH_SIZE = 64
EPOCHS = 15
N = len(train_imgs)

print("\n===== Training =====")
for epoch in range(EPOCHS):
    model.train()
    idx = np.random.permutation(N)
    total_loss, correct, total = 0.0, 0, 0
    
    for i in range(0, N, BATCH_SIZE):
        batch_imgs = torch.from_numpy(train_imgs[idx[i:i+BATCH_SIZE]]).float().to(device)
        batch_lbls = torch.from_numpy(train_lbls[idx[i:i+BATCH_SIZE]]).long().to(device)
        
        optimizer.zero_grad()
        all_logits, _, _, _ = model(batch_imgs, num_glimpses=3, threshold=0.70, return_all=True)
        
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
    all_logits, all_focuses, all_embs, all_info_maps = model(
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
plt.title(f"Feature Clustering (Final GRU State)\nTest Acc: {test_acc*100:.1f}%")
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('experiments/01_semantic_clustering/feature_clustering.png', dpi=130)
print("Saved: feature_clustering.png")
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

plt.suptitle("Focal Point Trajectories (빨간점=시작 → 노란점=이후)", fontsize=13)
plt.tight_layout()
plt.savefig('experiments/01_semantic_clustering/focal_trajectories.png', dpi=130)
print("Saved: focal_trajectories.png")
plt.close()

print("\n===== 완료 =====")
