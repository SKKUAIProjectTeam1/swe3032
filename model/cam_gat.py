"""
campus_grid_gnn_designer.py (V16-GAT2)
- scatter_reduce 제거 → sparse softmax 직접 구현 (버전 무관)
- Warmup + Gradient Clipping 유지
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

sys.path.append(os.getcwd())
try:
    from building_places import BUILDING_POLY
except ImportError:
    with open('building_places.txt', 'r') as f:
        exec(f.read(), globals())

W, H = 2223, 2056
RES = 100
N = RES * RES

env_map = Image.new('L', (W, H), 0)
draw = ImageDraw.Draw(env_map)
for pts in BUILDING_POLY.values():
    draw.polygon(pts, fill=1)
env_small = env_map.resize((RES, RES), resample=Image.NEAREST)
is_building = np.array(env_small).astype(float)

dist_out = distance_transform_edt(1 - is_building)
ridge_map = np.where((dist_out > 2) & (dist_out < 8), 1.0, 0.0)
ridge_tensor = torch.tensor(ridge_map.flatten(), dtype=torch.float)

edge_indices = []
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

edge_index = torch.tensor(edge_indices, dtype=torch.long).t()
num_edges = edge_index.shape[1]

xs = torch.arange(RES).repeat(RES).float() / RES
ys = torch.arange(RES).repeat_interleave(RES).float() / RES
node_feats = torch.stack([
    torch.tensor(is_building.flatten(), dtype=torch.float),
    ridge_tensor, xs, ys
], dim=1)  # (N, 4)


def sparse_softmax(e, dst, n):
    """
    scatter_reduce 없이 구현하는 안정적인 sparse softmax
    dst 기준으로 같은 노드에 들어오는 엣지들끼리 softmax
    """
    # 1) dst별 max 구하기 (numerical stability)
    e_max = torch.full((n,), float('-inf'))
    for i in range(e.shape[0]):
        d = dst[i].item()
        if e[i].item() > e_max[d].item():
            e_max[d] = e[i]
    # inf 처리
    e_max = torch.where(e_max == float('-inf'), torch.zeros(n), e_max)

    # 2) exp(e - max)
    e_exp = torch.exp(e - e_max[dst])

    # 3) dst별 합산
    e_sum = torch.zeros(n)
    e_sum.index_add_(0, dst, e_exp)

    # 4) 정규화
    alpha = e_exp / (e_sum[dst] + 1e-8)
    return alpha


class GATLayer(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.W = nn.Linear(in_dim, out_dim, bias=False)
        self.attn = nn.Linear(out_dim * 2, 1, bias=False)
        nn.init.xavier_uniform_(self.W.weight, gain=0.5)
        nn.init.xavier_uniform_(self.attn.weight, gain=0.5)

    def forward(self, x, edge_index):
        src, dst = edge_index[0], edge_index[1]
        h = self.W(x)  # (N, out_dim)

        attn_input = torch.cat([h[src], h[dst]], dim=1)
        e = F.leaky_relu(self.attn(attn_input).squeeze(), negative_slope=0.2)

        # ✅ scatter_reduce 없이 안정적인 softmax
        alpha = sparse_softmax(e, dst, N)

        out = torch.zeros(N, h.shape[1])
        out.index_add_(0, dst, alpha.unsqueeze(1) * h[src])
        return F.elu(out), alpha


class CivilEngineerGAT(nn.Module):
    def __init__(self, res):
        super().__init__()
        self.gat1 = GATLayer(4, 16)
        self.gat2 = GATLayer(16, 8)
        self.edge_predictor = nn.Sequential(
            nn.Linear(16, 8), nn.ReLU(), nn.Linear(8, 1)
        )
        self.boundary_indices = []
        for i in range(20, res - 20):
            self.boundary_indices.extend([i, (res-1)*res + i, i*res, i*res + (res-1)])
        self.boundary_indices = sorted(list(set(self.boundary_indices)))
        self.gate_logits = nn.Parameter(torch.randn(len(self.boundary_indices)) * 0.1)

    def forward(self, node_feats, edge_index):
        src, dst = edge_index[0], edge_index[1]
        h1, _ = self.gat1(node_feats, edge_index)
        h2, _ = self.gat2(h1, edge_index)
        edge_input = torch.cat([h2[src], h2[dst]], dim=1)
        edge_weights = torch.sigmoid(self.edge_predictor(edge_input).squeeze())
        return edge_weights, torch.sigmoid(self.gate_logits)


building_nodes = []
for bid, pts in BUILDING_POLY.items():
    p = np.mean(pts, axis=0) * (RES / W)
    building_nodes.append(max(0, min(N-1, int(p[1]) * RES + int(p[0]))))

model = CivilEngineerGAT(RES)
optimizer = optim.Adam(model.parameters(), lr=0.01)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10000, gamma=0.5)
is_building_tensor = torch.tensor(is_building.flatten(), dtype=torch.float)

WARMUP_STEPS = 3000
print("▶ V16-GAT2: sparse softmax 직접 구현 학습 시작 (50,000 steps)...")

for step in range(50001):
    optimizer.zero_grad()
    edge_weights, gate_weights = model(node_feats, edge_index)
    src, dst = edge_index[0], edge_index[1]

    collision_scale = min(1.0, step / WARMUP_STEPS) * 10000.0
    collision_mask = (is_building_tensor[src] > 0) | (is_building_tensor[dst] > 0)
    collision_loss = torch.sum(edge_weights[collision_mask] * collision_scale)

    ridge_score = (ridge_tensor[src] + ridge_tensor[dst]) / 2.0
    ridge_loss = torch.sum(edge_weights * (1.0 - ridge_score)) * 10.0

    node_strength = torch.zeros(N)
    node_strength.index_add_(0, src, edge_weights)
    node_strength.index_add_(0, dst, edge_weights)
    conn_loss = torch.sum(torch.pow(F.relu(2.0 - node_strength[building_nodes]), 2)) * 30.0

    sparsity_loss = torch.norm(edge_weights, p=1) * 0.1
    degree_loss = torch.sum(F.relu(node_strength - 3.5)) * 5.0
    gate_sum_loss = torch.pow(torch.sum(gate_weights) - 2.0, 2) * 500.0

    total_loss = collision_loss + ridge_loss + conn_loss + sparsity_loss + degree_loss + gate_sum_loss
    total_loss.backward()

    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()
    scheduler.step()

    if step % 5000 == 0:
        conn_count = (node_strength[building_nodes] > 1.5).sum().item()
        lr_now = scheduler.get_last_lr()[0]
        print(f"  Step {step:5d} | Loss: {total_loss.item():.2f} | Connected: {conn_count}/23 | Collisions: {collision_loss.item()/max(collision_scale,1):.4f} | LR: {lr_now:.5f}")

final_edge_weights = edge_weights.detach().numpy()
final_gate_weights = gate_weights.detach().numpy()

top_gate_local_indices = np.argsort(final_gate_weights)[-2:]
gate_node_indices = [model.boundary_indices[i] for i in top_gate_local_indices]
gates_coords = [((idx % RES) * (W/RES), (idx // RES) * (H/RES)) for idx in gate_node_indices]


def plot():
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

    plt.title("GNN Road Design V16-GAT2\n(Stable Sparse Softmax | Warmup | Gradient Clipping | 50,000 Steps)",
              color='white', fontsize=26, pad=50)
    plt.axis('off')
    ax.set_xlim(0, W)
    ax.set_ylim(H, 0)
    out_path = 'campus_v16_gat2_design.png'
    plt.savefig(out_path, dpi=200, bbox_inches='tight', facecolor='#050505')
    plt.close()
    print(f"\n✅ V16-GAT2 완료: {out_path}")


if __name__ == '__main__':
    plot()