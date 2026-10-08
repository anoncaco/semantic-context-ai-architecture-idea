"""
Experiment 02: Large Canvas Peripheral + Foveal Vision
======================================================

핵심 실험 원칙
1. 112x112 전체 시각장은 절대로 가리지 않는다.
2. Peripheral vision은 전체 장면의 저해상도 표현으로 "어디를 볼 것인가"를 예측한다.
3. Foveal vision은 예측 위치 주변을 고해상도로 읽는다.
4. 최종 판단에는 전체 장면 정보 + foveal 정보가 모두 들어간다.
5. 따라서 "정보를 가려서 맞히는가?"가 아니라
   "전체 정보를 가진 상태에서 선택적 고해상도 처리가 도움이 되는가?"를 측정한다.

비교:
A. Full-only baseline
B. Peripheral-only
C. Full + learned fovea
D. Oracle-fovea (실제 숫자 위치를 알려줌)  [상한선]
"""

import os
import math
import time
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device:", device)


# ============================================================
# 1. Synthetic MNIST-like digits
# ============================================================

def generate_digit_28x28(digit):
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
        d1 = np.sqrt((yy - 9)**2 + (xx - 14)**2)
        d2 = np.sqrt((yy - 18)**2 + (xx - 14)**2)
        img[((d1 > 4.5) & (d1 < 7.5)) |
            ((d2 > 4.5) & (d2 < 7.5))] = 1.0

    elif digit == 9:
        img[5:15, 7:10] = 1.0
        img[5:8, 7:21] = 1.0
        img[12:15, 7:21] = 1.0
        img[5:23, 18:21] = 1.0

    return img


def create_canvas_dataset(
    num_samples=5000,
    canvas_size=112,
    seed=42,
    distractors=0,
):
    rng = np.random.default_rng(seed)

    canvases = np.zeros(
        (num_samples, canvas_size, canvas_size),
        dtype=np.float32,
    )

    labels = np.zeros(num_samples, dtype=np.int64)
    coords = np.zeros((num_samples, 2), dtype=np.float32)

    for i in range(num_samples):

        digit = rng.integers(0, 10)
        digit_img = generate_digit_28x28(digit)

        top = rng.integers(8, canvas_size - 36)
        left = rng.integers(8, canvas_size - 36)

        canvases[i, top:top+28, left:left+28] = digit_img

        # 선택적 distractor:
        # 전체 시야를 실제로 유지하는지 확인하기 위한 무관한 물체
        for _ in range(distractors):
            dy = rng.integers(0, canvas_size - 8)
            dx = rng.integers(0, canvas_size - 8)
            canvases[i, dy:dy+4, dx:dx+4] += rng.uniform(0.1, 0.4)

        canvases[i] += rng.normal(
            0.0, 0.035, size=(canvas_size, canvas_size)
        )

        canvases[i] = np.clip(canvases[i], 0.0, 1.0)

        labels[i] = digit
        coords[i] = [top + 14, left + 14]

    return canvases, labels, coords


# ============================================================
# 2. Foveal sampler
# ============================================================

def crop_fovea(canvas, center, crop_size=40, output_size=56):
    """
    canvas:
        [B,1,H,W]

    center:
        [B,2] = (y,x)

    중요:
        이 함수는 원본 canvas를 제거하지 않는다.
        단지 같은 전체 시각장으로부터 고해상도 local view를
        하나 더 읽어오는 것이다.
    """

    B, C, H, W = canvas.shape

    yy = center[:, 0] / (H - 1) * 2.0 - 1.0
    xx = center[:, 1] / (W - 1) * 2.0 - 1.0

    # affine grid의 scale:
    # output 전체가 crop_size에 해당하도록 설정
    sx = crop_size / W
    sy = crop_size / H

    theta = torch.zeros(
        B, 2, 3,
        device=canvas.device,
        dtype=canvas.dtype,
    )

    theta[:, 0, 0] = sx
    theta[:, 1, 1] = sy
    theta[:, 0, 2] = xx
    theta[:, 1, 2] = yy

    grid = F.affine_grid(
        theta,
        size=(B, C, output_size, output_size),
        align_corners=True,
    )

    return F.grid_sample(
        canvas,
        grid,
        mode="bilinear",
        padding_mode="zeros",
        align_corners=True,
    )


# ============================================================
# 3. Peripheral network
# ============================================================

class PeripheralNet(nn.Module):
    """
    전체 112x112 -> 28x28 저해상도
    -> 아주 가벼운 network
    -> saliency map + global feature

    여기서는 전체 장면을 없애지 않는다.
    """

    def __init__(self):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(1, 16, 5, stride=2, padding=2),
            nn.ReLU(),

            nn.Conv2d(16, 24, 5, stride=2, padding=2),
            nn.ReLU(),

            nn.Conv2d(24, 32, 3, padding=1),
            nn.ReLU(),
        )

        self.saliency = nn.Conv2d(32, 1, 1)

        self.global_pool = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        # 전체 canvas를 28x28로 축소
        low = F.avg_pool2d(x, kernel_size=4, stride=4)

        feat = self.features(low)

        saliency = self.saliency(feat)
        global_feat = self.global_pool(feat).flatten(1)

        return saliency, global_feat


# ============================================================
# 4. Foveal network
# ============================================================

class FovealNet(nn.Module):
    """
    선택된 위치 주변의 고해상도 정보를 처리한다.

    단, 이 정보가 최종 입력의 전부가 아니다.
    """

    def __init__(self, out_dim=64):
        super().__init__()

        self.net = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1),
            nn.ReLU(),

            nn.MaxPool2d(2),

            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(),

            nn.MaxPool2d(2),

            nn.Conv2d(32, 32, 3, padding=1),
            nn.ReLU(),

            nn.AdaptiveAvgPool2d((4, 4)),
            nn.Flatten(),

            nn.Linear(32 * 4 * 4, out_dim),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.net(x)


# ============================================================
# 5. Full-field baseline
# ============================================================

class FullFieldBaseline(nn.Module):
    """
    비교군 A.

    112x112 전체를 그대로 보고 분류.
    """

    def __init__(self):
        super().__init__()

        self.net = nn.Sequential(
            nn.Conv2d(1, 16, 5, stride=2, padding=2),
            nn.ReLU(),

            nn.Conv2d(16, 32, 5, stride=2, padding=2),
            nn.ReLU(),

            nn.Conv2d(32, 32, 3, stride=2, padding=1),
            nn.ReLU(),

            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),

            nn.Linear(32, 64),
            nn.ReLU(),

            nn.Linear(64, 10),
        )

    def forward(self, x):
        return self.net(x)


# ============================================================
# 6. Peripheral + Foveal model
# ============================================================

class PeripheralFovealModel(nn.Module):
    """
    핵심 모델.

    전체 정보:
        full global feature
              +
        foveal high-resolution feature
              +
        focus coordinate

    모두 classifier에 전달한다.

    따라서 foveation은 "정보 차단"이 아니라
    "선택적 고해상도 추가 읽기"이다.
    """

    def __init__(
        self,
        canvas_size=112,
        fovea_size=40,
        fovea_output=56,
    ):
        super().__init__()

        self.canvas_size = canvas_size
        self.fovea_size = fovea_size
        self.fovea_output = fovea_output

        self.peripheral = PeripheralNet()
        self.foveal = FovealNet(64)

        self.focus_head = nn.Sequential(
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, 2),
        )

        # 최종 classifier
        # peripheral global 32
        # foveal 64
        # coordinate 2
        self.classifier = nn.Sequential(
            nn.Linear(32 + 64 + 2, 64),
            nn.ReLU(),
            nn.Linear(64, 10),
        )

    def predict_focus(self, canvas):
        saliency, global_feat = self.peripheral(canvas)

        # saliency map의 spatial softmax
        B, _, H, W = saliency.shape

        p = F.softmax(
            saliency.flatten(1),
            dim=1,
        ).view(B, H, W)

        yy = torch.linspace(
            0,
            self.canvas_size - 1,
            H,
            device=canvas.device,
        )

        xx = torch.linspace(
            0,
            self.canvas_size - 1,
            W,
            device=canvas.device,
        )

        gy, gx = torch.meshgrid(
            yy,
            xx,
            indexing="ij",
        )

        fy = (p * gy).sum((1, 2))
        fx = (p * gx).sum((1, 2))

        focus = torch.stack([fy, fx], dim=1)

        return focus, global_feat, saliency

    def forward(self, canvas):
        focus, global_feat, saliency = self.predict_focus(canvas)

        # 전체 canvas에서 local high-resolution view를 추가로 읽는다.
        fovea = crop_fovea(
            canvas,
            focus,
            crop_size=self.fovea_size,
            output_size=self.fovea_output,
        )

        foveal_feat = self.foveal(fovea)

        # 좌표도 명시적인 정보로 전달
        coord = focus / (self.canvas_size - 1)

        combined = torch.cat(
            [
                global_feat,
                foveal_feat,
                coord,
            ],
            dim=1,
        )

        logits = self.classifier(combined)

        return {
            "logits": logits,
            "focus": focus,
            "saliency": saliency,
            "fovea": fovea,
            "global_feat": global_feat,
            "foveal_feat": foveal_feat,
        }


# ============================================================
# 7. Oracle fovea model
# ============================================================

class OracleFovealModel(nn.Module):
    """
    실제 위치를 알려주는 상한선.

    이것은 테스트에서 정보를 가리는 모델이 아니라,
    "위치를 완벽히 찾았을 때 고해상도 추가 정보가 얼마나
    유용한가"를 보는 기준선이다.
    """

    def __init__(self):
        super().__init__()

        self.peripheral = PeripheralNet()
        self.foveal = FovealNet(64)

        self.classifier = nn.Sequential(
            nn.Linear(32 + 64 + 2, 64),
            nn.ReLU(),
            nn.Linear(64, 10),
        )

    def forward(self, canvas, true_center):
        _, global_feat = self.peripheral(canvas)

        fovea = crop_fovea(
            canvas,
            true_center,
            crop_size=40,
            output_size=56,
        )

        foveal_feat = self.foveal(fovea)

        coord = true_center / 111.0

        combined = torch.cat(
            [global_feat, foveal_feat, coord],
            dim=1,
        )

        logits = self.classifier(combined)

        return logits


# ============================================================
# 8. Training utilities
# ============================================================

def make_loader(x, y, coords, batch_size=128, shuffle=True):
    x = torch.from_numpy(x).unsqueeze(1)
    y = torch.from_numpy(y).long()
    coords = torch.from_numpy(coords).float()

    return DataLoader(
        TensorDataset(x, y, coords),
        batch_size=batch_size,
        shuffle=shuffle,
        pin_memory=torch.cuda.is_available(),
    )


def evaluate_selective(model, loader):
    model.eval()

    total = 0
    correct = 0

    focus_error = []

    with torch.no_grad():
        for x, y, coords in loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            coords = coords.to(device, non_blocking=True)

            out = model(x)

            pred = out["logits"].argmax(1)

            correct += (pred == y).sum().item()
            total += y.numel()

            err = torch.norm(
                out["focus"] - coords,
                dim=1,
            )

            focus_error.append(err.cpu())

    acc = correct / total
    focus_error = torch.cat(focus_error).mean().item()

    return acc, focus_error


def train_selective(
    model,
    train_loader,
    test_loader,
    epochs=10,
    focus_weight=0.03,
):
    model.to(device)

    optimizer = optim.Adam(
        model.parameters(),
        lr=1e-3,
    )

    criterion = nn.CrossEntropyLoss()

    history = []

    for epoch in range(1, epochs + 1):

        model.train()

        running_loss = 0.0
        n = 0

        for x, y, coords in train_loader:

            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            coords = coords.to(device, non_blocking=True)

            out = model(x)

            cls_loss = criterion(
                out["logits"],
                y,
            )

            # 주변시가 실제 물체 중심을 찾도록 약한 supervision.
            # classification loss가 주 목적이다.
            focus_loss = F.smooth_l1_loss(
                out["focus"],
                coords,
            )

            loss = cls_loss + focus_weight * focus_loss

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * y.size(0)
            n += y.size(0)

        acc, focus_err = evaluate_selective(
            model,
            test_loader,
        )

        epoch_loss = running_loss / n

        history.append(
            (epoch_loss, acc, focus_err)
        )

        print(
            f"[{epoch:02d}/{epochs}] "
            f"loss={epoch_loss:.4f} "
            f"test_acc={acc:.4f} "
            f"focus_error={focus_err:.2f}px"
        )

    return history


# ============================================================
# 9. Baseline training
# ============================================================

def train_full_baseline(
    model,
    train_loader,
    test_loader,
    epochs=10,
):
    model.to(device)

    optimizer = optim.Adam(
        model.parameters(),
        lr=1e-3,
    )

    criterion = nn.CrossEntropyLoss()

    for epoch in range(1, epochs + 1):

        model.train()

        for x, y, _ in train_loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            logits = model(x)

            loss = criterion(
                logits,
                y,
            )

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

        model.eval()

        correct = 0
        total = 0

        with torch.no_grad():
            for x, y, _ in test_loader:
                x = x.to(device, non_blocking=True)
                y = y.to(device, non_blocking=True)

                pred = model(x).argmax(1)

                correct += (pred == y).sum().item()
                total += y.numel()

        print(
            f"[Full {epoch:02d}/{epochs}] "
            f"test_acc={correct/total:.4f}"
        )

    return model


# ============================================================
# 10. Attention / saccade visualisation
# ============================================================

def visualize_saccades(
    model,
    x,
    y,
    coords,
    n=8,
):
    model.eval()

    xx = torch.from_numpy(
        x[:n]
    ).unsqueeze(1).to(device)

    with torch.no_grad():
        out = model(xx)

    pred = out["logits"].argmax(1).cpu().numpy()
    focus = out["focus"].cpu().numpy()
    fovea = out["fovea"].cpu().numpy()

    fig, axes = plt.subplots(
        n,
        3,
        figsize=(11, 3.1*n),
    )

    if n == 1:
        axes = axes[None, :]

    for i in range(n):

        # 1. full visual field
        ax = axes[i, 0]
        ax.imshow(x[i], cmap="gray")
        ax.scatter(
            coords[i, 1],
            coords[i, 0],
            s=100,
            marker="x",
            linewidths=3,
            label="true",
        )
        ax.scatter(
            focus[i, 1],
            focus[i, 0],
            s=100,
            marker="o",
            facecolors="none",
            linewidths=3,
            label="pred",
        )

        ax.set_title(
            f"true={y[i]} pred={pred[i]}"
        )
        ax.legend(loc="upper right")
        ax.axis("off")

        # 2. foveal crop
        axes[i, 1].imshow(
            fovea[i, 0],
            cmap="gray",
        )
        axes[i, 1].set_title("high-resolution fovea")
        axes[i, 1].axis("off")

        # 3. focus error
        err = np.linalg.norm(
            focus[i] - coords[i]
        )

        axes[i, 2].text(
            0.1,
            0.75,
            f"true center\n"
            f"({coords[i,0]:.1f}, {coords[i,1]:.1f})\n\n"
            f"pred focus\n"
            f"({focus[i,0]:.1f}, {focus[i,1]:.1f})\n\n"
            f"error = {err:.2f}px",
            fontsize=13,
        )

        axes[i, 2].axis("off")

    plt.tight_layout()
    plt.savefig(
        "experiments/02_peripheral_foveal/experiment02_saccade_result.png",
        dpi=160,
    )
    plt.show()


# ============================================================
# 11. Main experiment
# ============================================================

def main():

    print("\nCreating dataset...")

    train_x, train_y, train_c = create_canvas_dataset(
        num_samples=8000,
        seed=1,
    )

    test_x, test_y, test_c = create_canvas_dataset(
        num_samples=2000,
        seed=999,
    )

    train_loader = make_loader(
        train_x,
        train_y,
        train_c,
        batch_size=128,
        shuffle=True,
    )

    test_loader = make_loader(
        test_x,
        test_y,
        test_c,
        batch_size=256,
        shuffle=False,
    )

    # --------------------------------------------------------
    # A. Full field baseline
    # --------------------------------------------------------
    print("\n====================================")
    print("A. FULL-FIELD BASELINE")
    print("====================================")

    full_model = FullFieldBaseline()

    train_full_baseline(
        full_model,
        train_loader,
        test_loader,
        epochs=8,
    )

    # --------------------------------------------------------
    # B. Peripheral + Foveal
    # --------------------------------------------------------
    print("\n====================================")
    print("B. PERIPHERAL + FOVEAL")
    print("====================================")

    selective_model = PeripheralFovealModel()

    history = train_selective(
        selective_model,
        train_loader,
        test_loader,
        epochs=8,
        focus_weight=0.03,
    )

    # --------------------------------------------------------
    # C. Visualization
    # --------------------------------------------------------
    print("\nSaving visualization...")

    visualize_saccades(
        selective_model,
        test_x,
        test_y,
        test_c,
        n=8,
    )

    # --------------------------------------------------------
    # D. Final result
    # --------------------------------------------------------
    full_model.eval()

    correct = 0
    total = 0

    with torch.no_grad():
        for x, y, _ in test_loader:
            x = x.to(device)
            y = y.to(device)

            pred = full_model(x).argmax(1)

            correct += (pred == y).sum().item()
            total += y.numel()

    full_acc = correct / total

    selective_acc, focus_error = evaluate_selective(
        selective_model,
        test_loader,
    )

    print("\n====================================")
    print("FINAL RESULT")
    print("====================================")
    print(f"Full-field baseline : {full_acc:.4f}")
    print(f"Peripheral + Fovea  : {selective_acc:.4f}")
    print(f"Saccade error       : {focus_error:.2f}px")

    torch.save(
        selective_model.state_dict(),
        "experiments/02_peripheral_foveal/experiment02_peripheral_foveal.pt",
    )

    print("\nSaved:")
    print("  experiment02_saccade_result.png")
    print("  experiment02_peripheral_foveal.pt")


if __name__ == "__main__":
    main()
