"""
campus_grid_gnn_designer.py (V21 - Smart Quad-Gate Civil Engineer)
- 100x100 고해상도, 50,000 Steps
- Gate Diversity: 관문끼리 서로 밀어내어 중첩 방지 (3, 4번 겹침 해결)
- Proximity Bias: 관문이 건물군과 너무 멀어지지 않도록 유도 (1, 2번 이탈 해결)
- V16 Ridge Logic: 건물 사이 정중앙을 타는 고퀄리티 내부 도로 유지
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

sys.stdout.reconfigure(encoding='utf-8')

# 데이터 로드
sys.path.append(os.getcwd())
data_path = sys.argv[1] if len(sys.argv) > 1 else 'building_places.txt'
with open(data_path, 'r', encoding='utf-8') as f:
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

# [제약 1] Ridge Map (V16 고수)
dist_out = distance_transform_edt(1 - is_building)
ridge_map = np.where((dist_out > 2) & (dist_out < 8), 1.0, 0.0)
ridge_tensor = torch.tensor(ridge_map.flatten(), dtype=torch.float)

# 2. 8방향 격자 엣지
edge_indices = []
for y in range(RES):
    for x in range(RES):
        curr = y * RES + x
        for dy in [-1, 0, 1]:
            for dx in [-1, 0, 1]:
                if dy == 0 and dx == 0: continue
                nx, ny = x + dx, y + dy
                if 0 <= nx < RES and 0 <= ny < RES:
                    edge_indices.append([curr, ny * RES + nx])

edge_index = torch.tensor(edge_indices, dtype=torch.long).t()
num_edges = edge_index.shape[1]

# 3. 모델 정의
class SmartQuadGateGNN(nn.Module):
    def __init__(self, n_edges, res):
        super().__init__()
        self.edge_logits = nn.Parameter(torch.randn(n_edges) * 0.01 - 2.0)
        # 관문 후보지 (테두리 중앙부)
        self.boundary_indices = []
        for i in range(20, res-20):
            self.boundary_indices.extend([i, (res-1)*res + i, i*res, i*res + (res-1)])
        self.boundary_indices = sorted(list(set(self.boundary_indices)))
        self.gate_logits = nn.Parameter(torch.randn(len(self.boundary_indices)) * 0.1)
        
    def forward(self):
        return torch.sigmoid(self.edge_logits), torch.sigmoid(self.gate_logits)

# 4. 건물 위치 및 중심점
building_nodes = []
bld_centers = []
for bid, pts in BUILDING_POLY.items():
    p = np.mean(pts, axis=0) * (RES / W)
    building_nodes.append(max(0, min(RES*RES-1, int(p[1]) * RES + int(p[0]))))
    bld_centers.append(p)
bld_avg_center = np.mean(bld_centers, axis=0)

# ── 학습 ─────────────────────────────────────────────────────────────────────
model = SmartQuadGateGNN(num_edges, RES)
optimizer = optim.Adam(model.parameters(), lr=0.05)
is_building_tensor = torch.tensor(is_building.flatten(), dtype=torch.float)

print(f"▶ V21: 관문 중첩 방지 및 이탈 차단 로직 적용 학습 시작...")

for step in range(50001):
    optimizer.zero_grad()
    edge_weights, gate_weights = model()
    src, dst = edge_index[0], edge_index[1]
    
    # [Loss 1] Collision (건물 보호)
    collision_mask = (is_building_tensor[src] > 0) | (is_building_tensor[dst] > 0)
    collision_loss = torch.sum(edge_weights[collision_mask] * 20000.0)
    
    # [Loss 2] Connectivity (V16 고수)
    node_strength = torch.zeros(RES*RES)
    node_strength.index_add_(0, src, edge_weights)
    node_strength.index_add_(0, dst, edge_weights)
    conn_loss = torch.sum(torch.pow(F.relu(2.2 - node_strength[building_nodes]), 2)) * 40.0
    
    # [Loss 3] Gate Optimization (V21 핵심)
    # 1) 개수 제약
    gate_sum_loss = torch.pow(torch.sum(gate_weights) - 4.0, 2) * 1000.0
    
    # 2) Diversity Loss: 관문 후보지들 간의 상호 거리 패널티
    # 가중치가 높은 노드들끼리 서로 밀어내도록 함 (간이 gravity/repulsion)
    gate_pos = []
    for idx in model.boundary_indices:
        y, x = divmod(idx, RES)
        gate_pos.append([x, y])
    gate_pos_t = torch.tensor(gate_pos, dtype=torch.float)
    dist_gates = torch.cdist(gate_pos_t, gate_pos_t)
    # 가까운 거리에 있는 문들의 가중치 곱에 패널티
    repulsion = torch.exp(-dist_gates / 15.0) # 15패치 이내면 강한 패널티
    diversity_loss = torch.sum(gate_weights.view(-1, 1) * gate_weights.view(1, -1) * repulsion) * 50.0
    
    # 3) Proximity Reward: 건물군 중심과 가까운 외곽 지점 선호
    bld_center_t = torch.tensor(bld_avg_center, dtype=torch.float)
    dist_to_buildings = torch.norm(gate_pos_t - bld_center_t, dim=1)
    proximity_loss = torch.sum(gate_weights * dist_to_buildings) * 2.0
    
    # 4) 도로 연결 보장
    boundary_strength = node_strength[model.boundary_indices]
    gate_link_loss = torch.sum(gate_weights * torch.pow(F.relu(2.5 - boundary_strength), 2)) * 100.0
    
    # [Loss 4] Ridge & Regularization
    ridge_score = (ridge_tensor[src] + ridge_tensor[dst]) / 2.0
    ridge_loss = torch.sum(edge_weights * (1.0 - ridge_score)) * 12.0
    sparsity_loss = torch.norm(edge_weights, p=1) * 0.15
    degree_loss = torch.sum(F.relu(node_strength - 3.5)) * 10.0

    total_loss = collision_loss + conn_loss + gate_sum_loss + diversity_loss + proximity_loss + gate_link_loss + ridge_loss + sparsity_loss + degree_loss
    
    total_loss.backward()
    optimizer.step()
    
    if step % 5000 == 0:
        print(f"  Step {step:5d} | Loss: {total_loss.item():.2f} | Gate-Repulsion: {diversity_loss.item():.2f}")

# ── 결과 도출 및 시각화 ────────────────────────────────────────────────────────
final_edge_weights = edge_weights.detach().numpy()
final_gate_weights = gate_weights.detach().numpy()

# 관문 선별 알고리즘 (Greedy Selection for Diversity)
gate_scores = final_gate_weights.copy()
selected_gate_indices = []
for _ in range(4):
    best_idx = np.argmax(gate_scores)
    selected_gate_indices.append(model.boundary_indices[best_idx])
    # 선택된 지점 주변 25패치 무력화 (중첩 방지)
    y1, x1 = divmod(model.boundary_indices[best_idx], RES)
    for i in range(len(model.boundary_indices)):
        y2, x2 = divmod(model.boundary_indices[i], RES)
        if np.sqrt((x1-x2)**2 + (y1-y2)**2) < 25:
            gate_scores[i] = -1.0

gates_coords = [( (idx % RES) * (W/RES), (idx // RES) * (H/RES) ) for idx in selected_gate_indices]

def plot_v21():
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
            if is_building_tensor[s] > 0 or is_building_tensor[d] > 0: continue
            y1, x1 = divmod(s, RES); y2, x2 = divmod(d, RES)
            rx1, ry1 = x1 * (W/RES), y1 * (H/RES); rx2, ry2 = x2 * (W/RES), y2 * (H/RES)
            lw = (w - threshold) / (1.0 - threshold) * 12 + 2
            ax.plot([rx1, rx2], [ry1, ry2], color='#fff200', alpha=0.9, lw=lw, zorder=5, solid_capstyle='round')

    # 관문 (번호 매김)
    for i, (gx, gy) in enumerate(gates_coords):
        ax.scatter(gx, gy, s=1500, c='#ff3838', marker='*', edgecolors='white', lw=3, zorder=20)
        ax.text(gx, gy-100, f"GATE {i+1}", color='#ff3838', ha='center', fontweight='bold', fontsize=20, 
                bbox=dict(fc='black', alpha=0.8, ec='white'))

    plt.title("GNN Smart Quad-Gate Campus Design V21 (50,000 Steps)\n(Diversity & Proximity Optimized | No Overlap | Ridge-Aligned)", 
              color='white', fontsize=26, pad=50)
    plt.axis('off')
    ax.set_xlim(0, W); ax.set_ylim(H, 0)
    
    campus_name = os.path.basename(data_path).replace('_building_places.txt', '').replace('.txt', '')
    out_path = f'output/size_test/campus_v21_{campus_name}.png'
    plt.savefig(out_path, dpi=200, bbox_inches='tight', facecolor='#050505')
    plt.close()
    print(f"\n✅ V21 최종 도로 설계 완료: {out_path}")

if __name__ == '__main__':
    plot_v21()
