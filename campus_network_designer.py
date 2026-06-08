"""
campus_network_designer.py
GNN을 활용한 최적 도로망 및 관문(Gate) 위치 설계.
데이터 없이 오직 건물 위치와 기하학적 효율성만 따집니다.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os
import sys

sys.path.append(os.getcwd())
from campus_graph import BUILDINGS

# ── 설정 ──────────────────────────────────────────────────────────────────
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
MAP_PATH = '/home/sean429/swe3032/maps/카카오맵확대.png'
H, W = 2056, 2223  # 이미지 해상도

# 허브 지정
HUBS = ['03', '48']  # 학생회관, 도서관
NON_HUBS = [b for b in BUILDINGS.keys() if b not in HUBS]
ALL_BLDS = list(BUILDINGS.keys())
N_BLDS = len(ALL_BLDS)

# 관문(Gate) 후보지 생성 (캠퍼스 경계 8곳)
# x: 0~W, y: 0~H (이미지 좌표계, y는 위가 0)
GATE_CANDIDATES = {
    'G_North': (1100, 100),  'G_South': (1500, 1950),
    'G_West': (100, 1200),   'G_East': (2100, 1200),
    'G_NW': (300, 300),      'G_NE': (1900, 300),
    'G_SW': (300, 1800),     'G_SE': (1900, 1800)
}
GATE_IDS = list(GATE_CANDIDATES.keys())
N_GATES = len(GATE_IDS)

# 전체 노드: 건물(23) + 관문 후보(8) = 31개
NODE_POS = []
for b in ALL_BLDS:
    NODE_POS.append([BUILDINGS[b]['campus_x'], BUILDINGS[b]['campus_y']])
for g in GATE_IDS:
    NODE_POS.append([GATE_CANDIDATES[g][0], GATE_CANDIDATES[g][1]])

NODE_POS = torch.tensor(NODE_POS, dtype=torch.float).to(DEVICE)
N_TOTAL = N_BLDS + N_GATES

# ── 모델 정의 ─────────────────────────────────────────────────────────────────
class NetworkDesigner(nn.Module):
    def __init__(self, n_nodes):
        super().__init__()
        # 모든 쌍 간의 연결 가중치 (Logits)
        # 대칭 행렬을 위해 상삼각 행렬만 사용하거나, 전체를 학습 후 대칭화
        self.adj_logits = nn.Parameter(torch.randn(n_nodes, n_nodes) * 0.01)
        
    def get_adj(self):
        # 대칭화 및 Sigmoid를 통한 연결 확률 (0~1)
        adj = (self.adj_logits + self.adj_logits.t()) / 2
        return torch.sigmoid(adj)

def calculate_loss(adj, node_pos):
    """
    Loss = 건설 비용(거리 합) + 이동 비효율성(모든 노드 간 경로 거리)
    """
    # 1. 건설 비용 (Total Length)
    # dist_matrix: (N, N)
    dist_matrix = torch.cdist(node_pos, node_pos)
    cost_loss = torch.sum(adj * dist_matrix)
    
    # 2. 이동 비효율성 (Efficiency)
    # Floyd-Warshall의 미분 가능한 근사치를 사용하거나, 
    # 여기서는 단순화를 위해 adj를 가중치로 한 노드 간 거리의 역수를 활용
    # 실제로는 '강한 연결'이 있는 곳으로의 거리가 짧아야 함
    
    # 가상의 경로 손실: 모든 건물 쌍 (i, j)에 대해
    # 효율성 = 1 / (adj_ij + epsilon) * dist_ij  <- 이게 작아야 함
    # 특히 허브(03, 48)와 연결된 경로는 더 가중치를 줌
    
    # 허브 인덱스
    hub_idx = [ALL_BLDS.index(h) for h in HUBS]
    
    # 전역 효율성 (Soft Shortest Path Proxy)
    # adj가 높을수록 해당 경로의 '저항'이 낮다고 가정
    resistance = 1.0 / (adj + 1e-3)
    eff_loss = torch.mean(resistance * dist_matrix)
    
    # 허브 가중치 적용
    hub_eff = 0
    for h in hub_idx:
        hub_eff += torch.mean(resistance[h, :N_BLDS] * dist_matrix[h, :N_BLDS])
    
    # 관문 제약: 관문 후보지 8개 중 '정문/후문' 2개만 활성화되도록 유도 (Sparsity)
    # 관문 노드들의 총 연결 강도를 계산
    gate_strengths = torch.sum(adj[N_BLDS:, :N_BLDS], dim=1) # 각 관문이 건물들과 얼마나 연결되는가
    gate_loss = torch.norm(gate_strengths, p=1) # 관문 사용 최소화
    
    # Sparsity (불필요한 도로 억제 - "슈퍼도로 ㄴㄴ")
    sparsity_loss = torch.norm(adj, p=1)

    return 0.001 * cost_loss + 1.0 * eff_loss + 5.0 * hub_eff + 0.1 * sparsity_loss + 0.5 * gate_loss

# ── 학습 ─────────────────────────────────────────────────────────────────────
model = NetworkDesigner(N_TOTAL).to(DEVICE)
optimizer = torch.optim.Adam(model.parameters(), lr=0.1)

print("▶ AI가 최적 도로망을 설계 중입니다 (약 500 step)...")
for step in range(501):
    optimizer.zero_grad()
    adj = model.get_adj()
    loss = calculate_loss(adj, NODE_POS)
    loss.backward()
    optimizer.step()
    
    if step % 100 == 0:
        print(f"  Step {step:4d} | Loss {loss.item():.4f}")

# ── 결과 분석 및 시각화 ────────────────────────────────────────────────────────
final_adj = model.get_adj().detach().cpu().numpy()
node_pos_np = NODE_POS.cpu().numpy()

# 관문 선정: 연결 강도가 가장 높은 상위 2개 후보지
gate_strengths = np.sum(final_adj[N_BLDS:, :N_BLDS], axis=1)
top_gate_indices = np.argsort(gate_strengths)[-2:]
chosen_gates = [GATE_IDS[i] for i in top_gate_indices]
print(f"\n✅ AI 추천 관문 위치:")
print(f"  - 정문 후보: {chosen_gates[-1]} {GATE_CANDIDATES[chosen_gates[-1]]}")
print(f"  - 후문 후보: {chosen_gates[-2]} {GATE_CANDIDATES[chosen_gates[-2]]}")

def plot_design():
    img = np.array(Image.open(MAP_PATH))
    fig, ax = plt.subplots(figsize=(15, 14))
    ax.imshow(img)
    
    # 1. 도로망 그리기 (가중치 상위 엣지들)
    # 모든 쌍을 그리면 너무 지저분하므로 임계값 설정
    threshold = np.percentile(final_adj, 98) # 상위 2% 엣지만 표시
    for i in range(N_TOTAL):
        for j in range(i + 1, N_TOTAL):
            w = final_adj[i, j]
            if w > threshold:
                p1, p2 = node_pos_np[i], node_pos_np[j]
                lw = (w - threshold) / (1.0 - threshold) * 5 + 0.5
                alpha = min(1.0, lw / 5)
                color = 'blue' if i < N_BLDS and j < N_BLDS else 'orange'
                ax.plot([p1[0], p2[0]], [p1[1], p2[1]], color=color, alpha=alpha, lw=lw, zorder=2)

    # 2. 건물 표시
    for i, bld in enumerate(ALL_BLDS):
        x, y = node_pos_np[i]
        is_hub = bld in HUBS
        color = 'red' if is_hub else 'black'
        size = 100 if is_hub else 40
        ax.scatter(x, y, s=size, c=color, edgecolors='white', zorder=5)
        ax.text(x + 15, y, bld, fontsize=9, fontweight='bold' if is_hub else 'normal',
                color=color, bbox=dict(fc='white', alpha=0.7, ec='none', pad=1), zorder=6)

    # 3. 관문 표시
    for i, gid in enumerate(GATE_IDS):
        idx = N_BLDS + i
        x, y = node_pos_np[idx]
        is_chosen = gid in chosen_gates
        if is_chosen:
            label = "MAIN GATE" if gid == chosen_gates[-1] else "BACK GATE"
            ax.scatter(x, y, s=250, c='yellow', edgecolors='red', marker='D', zorder=7)
            ax.text(x, y - 40, label, ha='center', fontsize=12, fontweight='bold', color='red',
                    bbox=dict(fc='yellow', alpha=0.9, ec='red'), zorder=8)
        else:
            ax.scatter(x, y, s=50, c='orange', alpha=0.3, zorder=3)

    plt.title("AI Designed Optimal Campus Network & Gates\n(Red: Hubs, Blue: Internal Roads, Orange: Gate Access)", fontsize=18)
    plt.axis('off')
    out_path = 'campus_ai_design_result.png'
    plt.savefig(out_path, dpi=120, bbox_inches='tight')
    plt.close()
    print(f"\n✅ 설계 도면 저장 완료: {out_path}")

if __name__ == '__main__':
    plot_design()
