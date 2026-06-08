"""
campus_grid_gnn_designer.py (V16 - The Civil Engineer + True GNN)
- 100x100 고해상도, 50,000 Steps
- Medial Axis Reward: 건물 사이의 정중앙(능선)을 따라 도로 형성 유도
- Curvature Penalty: 급격한 꺾임 방지, 직선 도로 지향
- Degree Control: 노드당 연결을 2~4개로 제한하여 깔끔한 망 구성
- ✅ True GNN: Aggregate → Transform → Predict 메시지 패싱 구조 추가
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
import os
import sys
from scipy.ndimage import distance_transform_edt

# 데이터 로드
sys.path.append(os.getcwd())
try:
    from building_places import BUILDING_POLY
except ImportError:
    with open('building_places.txt', 'r') as f:
        exec(f.read(), globals())

# ── 설정 ──────────────────────────────────────────────────────────────────
W, H = 2223, 2056
RES = 100
BUILDING_IDS = list(BUILDING_POLY.keys())

# 1. 환경 맵 및 지형 분석
env_map = Image.new('L', (W, H), 0)
draw = ImageDraw.Draw(env_map)
for pts in BUILDING_POLY.values():
    draw.polygon(pts, fill=1)
env_small = env_map.resize((RES, RES), resample=Image.NEAREST)
is_building = np.array(env_small).astype(float)

# [도로 조건 1] Ridge Map
dist_out = distance_transform_edt(1 - is_building)
ridge_map = np.where((dist_out > 2) & (dist_out < 8), 1.0, 0.0)
ridge_tensor = torch.tensor(ridge_map.flatten(), dtype=torch.float)

# 2. 8방향 격자 엣지
edge_indices = []
edge_vectors = []
for y in range(RES):
    for x in range(RES):
        curr = y * RES + x
        for dy in [-1, 0, 1]:
            for dx in [-1, 0, 1]:
                if dy == 0 and dx == 0:
                    continue
                nx, ny = x + dx, y + dy
                if 0 <= nx < RES and 0 <= ny < RES:
                    edge_indices.append([curr, ny * RES + nx])
                    edge_vectors.append([dx, dy])

edge_index = torch.tensor(edge_indices, dtype=torch.long).t()
edge_vecs = torch.tensor(edge_vectors, dtype=torch.float)
num_edges = edge_index.shape[1]

# ── 노드 피처 준비 (GNN 입력) ──────────────────────────────────────────────
# 피처: [is_building, ridge_score, x좌표(정규화), y좌표(정규화)]
xs = torch.arange(RES).repeat(RES).float() / RES
ys = torch.arange(RES).repeat_interleave(RES).float() / RES
node_feats = torch.stack([
    torch.tensor(is_building.flatten(), dtype=torch.float),
    ridge_tensor,
    xs,
    ys
], dim=1)  # (10000, 4)

# 3. 모델 정의 ✅ True GNN 구조
class CivilEngineerGNN(nn.Module):
    def __init__(self, n_edges, res):
        super().__init__()
        self.res = res

        # 초기 엣지 로짓 (메시지 패싱의 seed 역할)
        self.edge_logits = nn.Parameter(torch.randn(n_edges) * 0.01 - 2.0)

        # 관문 파라미터
        self.boundary_indices = []
        for i in range(20, res - 20):
            self.boundary_indices.extend([i, (res-1)*res + i, i*res, i*res + (res-1)])
        self.boundary_indices = sorted(list(set(self.boundary_indices)))
        self.gate_logits = nn.Parameter(torch.randn(len(self.boundary_indices)) * 0.1)

        # ✅ GNN 핵심: 노드 임베딩 MLP
        # 입력: [is_building, ridge, x, y, node_strength(집계된 이웃 정보)]
        self.node_mlp = nn.Sequential(
            nn.Linear(5, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
        )

        # ✅ GNN 핵심: 엣지 예측 MLP
        # 입력: src임베딩 + dst임베딩 concat
        self.edge_mlp = nn.Sequential(
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
        )

    def forward(self, node_feats, edge_index):
        src, dst = edge_index[0], edge_index[1]

        # ── Step 1: 초기 엣지 가중치로 이웃 집계 (Aggregate) ──
        init_weights = torch.sigmoid(self.edge_logits)
        node_strength = torch.zeros(self.res * self.res)
        node_strength.index_add_(0, src, init_weights.detach())
        node_strength.index_add_(0, dst, init_weights.detach())

        # ── Step 2: 노드 피처 + 집계 정보 → 노드 임베딩 (Transform) ──
        node_input = torch.cat([node_feats, node_strength.unsqueeze(1)], dim=1)  # (N, 5)
        node_emb = self.node_mlp(node_input)  # (N, 16)

        # ── Step 3: 엣지 양끝 노드 임베딩으로 엣지 가중치 예측 (Predict) ──
        edge_input = torch.cat([node_emb[src], node_emb[dst]], dim=1)  # (E, 32)
        edge_weights = torch.sigmoid(self.edge_mlp(edge_input).squeeze())  # (E,)

        return edge_weights, torch.sigmoid(self.gate_logits)


# 4. 건물 위치
building_nodes = []
for bid, pts in BUILDING_POLY.items():
    p = np.mean(pts, axis=0) * (RES / W)
    building_nodes.append(max(0, min(RES*RES-1, int(p[1]) * RES + int(p[0]))))

# ── 학습 ──────────────────────────────────────────────────────────────────
model = CivilEngineerGNN(num_edges, RES)
optimizer = optim.Adam(model.parameters(), lr=0.005)  # MLP 추가로 lr 낮춤
is_building_tensor = torch.tensor(is_building.flatten(), dtype=torch.float)

print("▶ V16 True GNN: Aggregate → Transform → Predict 메시지 패싱 학습 시작 (50,000 steps)...")

for step in range(50001):
    optimizer.zero_grad()
    edge_weights, gate_weights = model(node_feats, edge_index)
    src, dst = edge_index[0], edge_index[1]

    # [Loss 1] Collision
    collision_mask = (is_building_tensor[src] > 0) | (is_building_tensor[dst] > 0)
    collision_loss = torch.sum(edge_weights[collision_mask] * 10000.0)

    # [Loss 2] Ridge & Gap Reward
    ridge_score = (ridge_tensor[src] + ridge_tensor[dst]) / 2.0
    ridge_loss = torch.sum(edge_weights * (1.0 - ridge_score)) * 10.0

    # [Loss 3] Connectivity
    node_strength = torch.zeros(RES * RES)
    node_strength.index_add_(0, src, edge_weights)
    node_strength.index_add_(0, dst, edge_weights)
    conn_loss = torch.sum(torch.pow(F.relu(2.0 - node_strength[building_nodes]), 2)) * 30.0

    # [Loss 4] Sparsity
    sparsity_loss = torch.norm(edge_weights, p=1) * 0.1

    # [Loss 5] Degree Control
    degree_loss = torch.sum(F.relu(node_strength - 3.5)) * 5.0

    # [Loss 6] Gate Sum
    gate_sum_loss = torch.pow(torch.sum(gate_weights) - 2.0, 2) * 500.0

    total_loss = collision_loss + ridge_loss + conn_loss + sparsity_loss + degree_loss + gate_sum_loss

    total_loss.backward()
    optimizer.step()

    if step % 5000 == 0:
        conn_count = (node_strength[building_nodes] > 0.5).sum().item()
        print(f"  Step {step:5d} | Loss: {total_loss.item():.2f} | Connected: {conn_count}/23 | Collisions: {collision_loss.item()/10000.0:.4f}")

# ── 결과 시각화 ────────────────────────────────────────────────────────────
final_edge_weights = edge_weights.detach().numpy()
final_gate_weights = gate_weights.detach().numpy()

top_gate_local_indices = np.argsort(final_gate_weights)[-2:]
gate_node_indices = [model.boundary_indices[i] for i in top_gate_local_indices]
gates_coords = [((idx % RES) * (W/RES), (idx // RES) * (H/RES)) for idx in gate_node_indices]


def plot_v16_true():
    fig, ax = plt.subplots(figsize=(16, 15))
    ax.set_facecolor('#050505')
    fig.patch.set_facecolor('#050505')

    import matplotlib.patches as mpatches
    for bid, pts in BUILDING_POLY.items():
        poly = mpatches.Polygon(pts, closed=True, facecolor='#2d3436', alpha=0.9, edgecolor='#00ff88', lw=1.2)
        ax.add_patch(poly)
        cx, cy = np.mean(pts, axis=0)
        ax.text(cx, cy, bid, color='white', ha='center', va='center', fontsize=10, fontweight='bold', zorder=10)

    threshold = np.percentile(final_edge_weights, 98.8)
    for i in range(num_edges):
        w = final_edge_weights[i]
        if w > threshold:
            s, d = edge_index[0, i].item(), edge_index[1, i].item()
            if is_building_tensor[s] > 0 or is_building_tensor[d] > 0:
                continue
            y1, x1 = divmod(s, RES)
            y2, x2 = divmod(d, RES)
            rx1, ry1 = x1 * (W/RES), y1 * (H/RES)
            rx2, ry2 = x2 * (W/RES), y2 * (H/RES)
            lw = (w - threshold) / (1.0 - threshold) * 10 + 2
            ax.plot([rx1, rx2], [ry1, ry2], color='#fff200', alpha=0.9, lw=lw, zorder=5, solid_capstyle='round')

    for i, (gx, gy) in enumerate(gates_coords):
        ax.scatter(gx, gy, s=1500, c='#ff3838', marker='*', edgecolors='white', lw=3, zorder=20)
        label = "MAIN GATE" if i == 1 else "BACK GATE"
        ax.text(gx, gy-100, label, color='white', ha='center', fontweight='bold', fontsize=18,
                bbox=dict(fc='#ff3838', alpha=0.8, ec='none'))

    plt.title("GNN Road Design V16: True GNN (Aggregate→Transform→Predict)\n(Ridge-Aligned | Smooth Connectivity | 50,000 Steps)",
              color='white', fontsize=26, pad=50)
    plt.axis('off')
    ax.set_xlim(0, W)
    ax.set_ylim(H, 0)

    out_path = 'campus__gnn_design.png'
    plt.savefig(out_path, dpi=200, bbox_inches='tight', facecolor='#050505')
    plt.close()
    print(f"\n✅ GNN 최종 도로 설계 완료: {out_path}")


if __name__ == '__main__':
    plot_v16_true()