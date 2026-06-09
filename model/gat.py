"""
campus_grid_gnn_designer.py (V16-GAT4)
- sparse matrix 곱으로 attention 구현 (버전 무관, GPU 최적화)
- h' = σ(A_attn · H · W) 형태의 정통 GAT
- Learnable node_embed + Warmup + Gradient Clipping
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
import signal
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

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"▶ Using device: {device}")

# 환경 맵
env_map = Image.new('L', (W, H), 0)
draw = ImageDraw.Draw(env_map)
for pts in BUILDING_POLY.values():
    draw.polygon(pts, fill=1)
env_small = env_map.resize((RES, RES), resample=Image.NEAREST)
is_building = np.array(env_small).astype(float)

dist_out = distance_transform_edt(1 - is_building)
ridge_map = np.where((dist_out > 2) & (dist_out < 8), 1.0, 0.0)
ridge_tensor = torch.tensor(ridge_map.flatten(), dtype=torch.float).to(device)

# 8방향 엣지
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

edge_index = torch.tensor(edge_indices, dtype=torch.long).t().to(device)
num_edges = edge_index.shape[1]

# 고정 노드 피처
xs = torch.arange(RES).repeat(RES).float() / RES
ys = torch.arange(RES).repeat_interleave(RES).float() / RES
static_feats = torch.stack([
    torch.tensor(is_building.flatten(), dtype=torch.float),
    ridge_tensor.cpu(), xs, ys
], dim=1).to(device)  # (N, 4)


# ── Sparse GAT 레이어 ──────────────────────────────────────────────────────
class SparseGATLayer(nn.Module):
    """
    h' = σ(A_attn · H · W)
    A_attn: attention 가중치로 만든 sparse adjacency matrix
    """
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.W = nn.Linear(in_dim, out_dim, bias=False)
        self.attn_src = nn.Linear(out_dim, 1, bias=False)  # src 어텐션
        self.attn_dst = nn.Linear(out_dim, 1, bias=False)  # dst 어텐션
        nn.init.xavier_uniform_(self.W.weight, gain=0.5)
        nn.init.xavier_uniform_(self.attn_src.weight, gain=0.5)
        nn.init.xavier_uniform_(self.attn_dst.weight, gain=0.5)

    def forward(self, x, edge_index):
        src, dst = edge_index[0], edge_index[1]

        # 1) 노드 변환
        h = self.W(x)  # (N, out_dim)

        # 2) attention score = LeakyReLU(a_src[src] + a_dst[dst])
        # 분리된 src/dst 어텐션 벡터 사용 (더 안정적)
        e = F.leaky_relu(
            self.attn_src(h)[src].squeeze() + self.attn_dst(h)[dst].squeeze(),
            negative_slope=0.2
        )  # (E,)

        # 3) ✅ sparse softmax: sparse matrix로 구현
        # edge마다 exp(e) 계산 후 sparse matrix 만들어서 row-wise normalize
        e_exp = torch.exp(e - e.max())  # numerical stability

        # sparse matrix (N×N): 값은 exp(e)
        sparse_A = torch.sparse_coo_tensor(
            edge_index,        # (2, E)
            e_exp,             # (E,)
            (N, N),
            device=device
        ).coalesce()

        # row sum으로 normalize → attention 계수
        row_sum = torch.sparse.sum(sparse_A, dim=1).to_dense() + 1e-8  # (N,)
        alpha = e_exp / row_sum[src]  # (E,) normalized

        # 4) ✅ sparse matmul로 집계: A_attn @ H
        sparse_A_norm = torch.sparse_coo_tensor(
            edge_index,
            alpha,
            (N, N),
            device=device
        ).coalesce()

        # sparse @ dense = dense (PyTorch 기본 지원)
        out = torch.sparse.mm(sparse_A_norm, h)  # (N, out_dim)

        return F.elu(out), alpha


# ── 모델 ───────────────────────────────────────────────────────────────────
class CivilEngineerGAT(nn.Module):
    def __init__(self, res):
        super().__init__()

        # 학습 가능한 노드 임베딩
        self.node_embed = nn.Parameter(torch.randn(N, 8) * 0.01)

        # Sparse GAT 2레이어: 입력 4+8=12 → 16 → 8
        self.gat1 = SparseGATLayer(in_dim=12, out_dim=16)
        self.gat2 = SparseGATLayer(in_dim=16, out_dim=8)

        # 엣지 예측
        self.edge_predictor = nn.Sequential(
            nn.Linear(16, 8),
            nn.ReLU(),
            nn.Linear(8, 1)
        )

        # 관문
        self.boundary_indices = []
        for i in range(20, res - 20):
            self.boundary_indices.extend([i, (res-1)*res + i, i*res, i*res + (res-1)])
        self.boundary_indices = sorted(list(set(self.boundary_indices)))
        self.gate_logits = nn.Parameter(torch.randn(len(self.boundary_indices)) * 0.1)

    def forward(self, static_feats, edge_index):
        src, dst = edge_index[0], edge_index[1]

        # 고정 피처 + 학습 임베딩
        node_feats = torch.cat([static_feats, self.node_embed], dim=1)  # (N, 12)

        # Sparse GAT 메시지 패싱
        h1, _ = self.gat1(node_feats, edge_index)   # (N, 16)
        h2, _ = self.gat2(h1, edge_index)            # (N, 8)

        # 엣지 예측
        edge_input = torch.cat([h2[src], h2[dst]], dim=1)  # (E, 16)
        edge_weights = torch.sigmoid(self.edge_predictor(edge_input).squeeze())

        return edge_weights, torch.sigmoid(self.gate_logits)


# 건물 위치
building_nodes = []
for bid, pts in BUILDING_POLY.items():
    p = np.mean(pts, axis=0) * (RES / W)
    building_nodes.append(max(0, min(N-1, int(p[1]) * RES + int(p[0]))))
building_nodes_t = torch.tensor(building_nodes, dtype=torch.long).to(device)

# ── 학습 ──────────────────────────────────────────────────────────────────
model = CivilEngineerGAT(RES).to(device)

# ✅ 파라미터 그룹 분리: node_embed는 빠르게, 나머지는 천천히
optimizer = optim.Adam([
    {'params': model.node_embed,             'lr': 0.1},   # node_embed 빠르게
    {'params': model.gate_logits,            'lr': 0.01},  # gate는 중간
    {'params': list(model.gat1.parameters()) +
               list(model.gat2.parameters()) +
               list(model.edge_predictor.parameters()), 'lr': 0.005}  # MLP 천천히
])
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=15000, gamma=0.3)
is_building_tensor = torch.tensor(is_building.flatten(), dtype=torch.float).to(device)

WARMUP_STEPS = 3000
print("▶ V16-GAT4: Sparse Matrix GAT 학습 시작 (50,000 steps)...")
print("   (Ctrl+C로 중단해도 그 시점 결과로 이미지 저장됨)")

interrupted = False
def handle_interrupt(sig, frame):
    global interrupted
    print("\n⚠️  중단 감지 — 현재 step 결과로 이미지 저장합니다...")
    interrupted = True
signal.signal(signal.SIGINT, handle_interrupt)

edge_weights, gate_weights = None, None

for step in range(50001):
    if interrupted:
        break

    optimizer.zero_grad()
    edge_weights, gate_weights = model(static_feats, edge_index)
    src, dst = edge_index[0], edge_index[1]

    collision_scale = min(1.0, step / WARMUP_STEPS) * 10000.0
    collision_mask = (is_building_tensor[src] > 0) | (is_building_tensor[dst] > 0)
    collision_loss = torch.sum(edge_weights[collision_mask] * collision_scale)

    ridge_score = (ridge_tensor[src] + ridge_tensor[dst]) / 2.0
    ridge_loss = torch.sum(edge_weights * (1.0 - ridge_score)) * 10.0

    node_strength = torch.zeros(N, device=device)
    node_strength.index_add_(0, src, edge_weights)
    node_strength.index_add_(0, dst, edge_weights)
    conn_loss = torch.sum(torch.pow(F.relu(2.0 - node_strength[building_nodes_t]), 2)) * 30.0

    sparsity_loss = torch.norm(edge_weights, p=1) * 0.1
    degree_loss = torch.sum(F.relu(node_strength - 3.5)) * 5.0
    gate_sum_loss = torch.pow(torch.sum(gate_weights) - 2.0, 2) * 500.0

    total_loss = collision_loss + ridge_loss + conn_loss + sparsity_loss + degree_loss + gate_sum_loss
    total_loss.backward()

    # ✅ Step 0 grad norm 진단
    if step == 0:
        print("\n── Gradient Norm 진단 (Step 0) ──")
        for name, param in model.named_parameters():
            if param.grad is not None:
                print(f"  {name:30s}: {param.grad.norm().item():.6f}")
            else:
                print(f"  {name:30s}: grad NONE ❌")
        print("──────────────────────────────────\n")

    # ✅ 파라미터 그룹별 clip: node_embed는 넉넉하게, MLP는 타이트하게
    torch.nn.utils.clip_grad_norm_(model.node_embed, max_norm=5.0)
    torch.nn.utils.clip_grad_norm_(model.gate_logits, max_norm=10.0)
    torch.nn.utils.clip_grad_norm_(
        list(model.gat1.parameters()) +
        list(model.gat2.parameters()) +
        list(model.edge_predictor.parameters()),
        max_norm=1.0
    )
    optimizer.step()
    scheduler.step()

    if step % 5000 == 0:
        conn_count = (node_strength[building_nodes_t] > 1.5).sum().item()
        lr_now = scheduler.get_last_lr()[0]
        print(f"  Step {step:5d} | Loss: {total_loss.item():.2f} | Connected: {conn_count}/23 | Collisions: {collision_loss.item()/max(collision_scale,1):.4f} | LR: {lr_now:.5f}")

if edge_weights is None:
    print("❌ 학습이 한 스텝도 안 돌았습니다.")
    sys.exit(1)

print(f"\n✅ 학습 완료 (step {step})")

# ── 결과 시각화 ────────────────────────────────────────────────────────────
final_edge_weights = edge_weights.detach().cpu().numpy()
final_gate_weights = gate_weights.detach().cpu().numpy()

top_gate_local_indices = np.argsort(final_gate_weights)[-2:]
gate_node_indices = [model.boundary_indices[i] for i in top_gate_local_indices]
gates_coords = [((idx % RES) * (W/RES), (idx // RES) * (H/RES)) for idx in gate_node_indices]
is_building_np = is_building_tensor.cpu().numpy()
ei_cpu = edge_index.cpu()


def plot():
    fig, ax = plt.subplots(figsize=(16, 15))
    ax.set_facecolor('#050505')
    fig.patch.set_facecolor('#050505')
    import matplotlib.patches as mpatches
    for bid, pts in BUILDING_POLY.items():
        poly = mpatches.Polygon(pts, closed=True, facecolor='#2d3436', alpha=0.9, edgecolor='#00ff88', lw=1.2)
        ax.add_patch(poly)
        cx, cy = np.mean(pts, axis=0)
        ax.text(cx, cy, bid, color='white', ha='center', va='center',
                fontsize=10, fontweight='bold', zorder=10)

    threshold = np.percentile(final_edge_weights, 98.8)
    print(f"  threshold: {threshold:.4f}, max: {final_edge_weights.max():.4f}, min: {final_edge_weights.min():.4f}")
    for i in range(num_edges):
        w = final_edge_weights[i]
        if w > threshold:
            s, d = ei_cpu[0, i].item(), ei_cpu[1, i].item()
            if is_building_np[s] > 0 or is_building_np[d] > 0:
                continue
            y1, x1 = divmod(s, RES)
            y2, x2 = divmod(d, RES)
            rx1, ry1 = x1 * (W/RES), y1 * (H/RES)
            rx2, ry2 = x2 * (W/RES), y2 * (H/RES)
            lw = (w - threshold) / (max(final_edge_weights.max() - threshold, 1e-8)) * 10 + 2
            ax.plot([rx1, rx2], [ry1, ry2], color='#fff200', alpha=0.9,
                    lw=lw, zorder=5, solid_capstyle='round')

    for i, (gx, gy) in enumerate(gates_coords):
        ax.scatter(gx, gy, s=1500, c='#ff3838', marker='*', edgecolors='white', lw=3, zorder=20)
        label = "MAIN GATE" if i == 1 else "BACK GATE"
        ax.text(gx, gy-100, label, color='white', ha='center', fontweight='bold',
                fontsize=18, bbox=dict(fc='#ff3838', alpha=0.8, ec='none'))

    plt.title("GNN Road Design V16-GAT4\n(Sparse Matrix GAT | Learnable Node Embed | 50,000 Steps)",
              color='white', fontsize=26, pad=50)
    plt.axis('off')
    ax.set_xlim(0, W)
    ax.set_ylim(H, 0)
    out_path = 'campus_v16_gat4_design.png'
    plt.savefig(out_path, dpi=200, bbox_inches='tight', facecolor='#050505')
    plt.close()
    print(f"✅ V16-GAT4 완료: {out_path}")


if __name__ == '__main__':
    plot()