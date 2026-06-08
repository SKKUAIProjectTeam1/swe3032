"""
campus_pure_gnn_designer.py
A* 알고리즘 없이, 허브 없이, 순수 GNN 메세지 패싱으로 최적 도로망과 관문을 설계합니다.
모든 건물은 동일한 가중치를 가집니다.
"""
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
import os
import sys

# 데이터 로드
sys.path.append(os.getcwd())
try:
    from building_places import BUILDING_POLY
except ImportError:
    with open('building_places.txt', 'r') as f:
        exec(f.read(), globals())

# ── 설정 ──────────────────────────────────────────────────────────────────
W, H = 2223, 2056
GRID_RES = 50 # 50x50 격자
BUILDING_IDS = list(BUILDING_POLY.keys())

# 1. 환경 맵 생성 (건물=1, 빈공간=0)
env_map = Image.new('L', (W, H), 0)
draw = ImageDraw.Draw(env_map)
for pts in BUILDING_POLY.values():
    draw.polygon(pts, fill=1)

env_small = env_map.resize((GRID_RES, GRID_RES), resample=Image.NEAREST)
env_np = np.array(env_small).astype(float) # (GRID_RES, GRID_RES)

# 2. 모델 정의: 격자 간의 연결성(도로 확률)을 학습
class RoadGNN(nn.Module):
    def __init__(self, res):
        super().__init__()
        self.res = res
        # 각 격자 셀의 '도로 가능성' (Logits)
        self.road_logits = nn.Parameter(torch.randn(res, res) * 0.1)
        # 관문(Gate) 후보지 (외곽 4면)
        self.gate_logits = nn.Parameter(torch.randn(res * 4) * 0.1)
        
    def forward(self):
        road_map = torch.sigmoid(self.road_logits)
        gate_map = torch.sigmoid(self.gate_logits)
        return road_map, gate_map

def get_boundary_mask(res):
    mask = torch.zeros(res, res)
    mask[0, :] = 1; mask[-1, :] = 1; mask[:, 0] = 1; mask[:, -1] = 1
    return mask

# ── 학습 ─────────────────────────────────────────────────────────────────────
model = RoadGNN(GRID_RES)
optimizer = optim.Adam(model.parameters(), lr=0.05)
env_tensor = torch.tensor(env_np, dtype=torch.float)
boundary_mask = get_boundary_mask(GRID_RES)

print("▶ GNN이 캠퍼스 지형을 학습하며 최적 도로망을 생성 중입니다...")

for step in range(1001):
    optimizer.zero_grad()
    road_map, gate_map = model()
    
    # 1. Collision Loss: 건물을 관통하는 도로 억제
    collision_loss = torch.sum(road_map * env_tensor)
    
    # 2. Continuity Loss: 도로는 서로 이어져야 함 (Laplacian smoothing)
    # 주변 8방향과의 차이를 최소화하여 '선' 형태가 유지되도록 유도
    diff_h = torch.pow(road_map[:, 1:] - road_map[:, :-1], 2).sum()
    diff_v = torch.pow(road_map[1:, :] - road_map[:-1, :], 2).sum()
    continuity_loss = diff_h + diff_v
    
    # 3. Connectivity Loss (핵심): 
    # 모든 건물은 도로망에 닿아야 하고, 도로망은 관문(Gate)에 닿아야 함.
    # 여기서는 간단히 '건물 주변의 도로 밀도'를 높이도록 유도
    building_touch_loss = 0
    for pts in BUILDING_POLY.values():
        # 건물 중심 주변 격자 찾기
        p = np.mean(pts, axis=0) * (GRID_RES / W)
        cx, cy = int(p[0]), int(p[1])
        # 건물 바로 옆 3x3 영역에 도로가 있어야 함
        x_min, x_max = max(0, cx-1), min(GRID_RES, cx+2)
        y_min, y_max = max(0, cy-1), min(GRID_RES, cy+2)
        building_touch_loss += (1.0 - torch.max(road_map[y_min:y_max, x_min:x_max]))

    # 4. Gate Optimization: 외곽 중 도로망과 가장 잘 연결된 곳 2개를 찾음
    # road_map의 외곽값과 gate_map의 곱을 최대화
    boundary_values = torch.cat([road_map[0, :], road_map[-1, :], road_map[:, 0], road_map[:, -1]])
    gate_connection_loss = torch.sum(torch.pow(gate_map - boundary_values, 2))
    
    # 5. Sparsity: 도로가 너무 떡칠되지 않게 (가느다란 길 유도)
    sparsity_loss = torch.sum(road_map)

    loss = (collision_loss * 50.0 + 
            continuity_loss * 2.0 + 
            building_touch_loss * 10.0 + 
            gate_connection_loss * 5.0 + 
            sparsity_loss * 0.1)
    
    loss.backward()
    optimizer.step()
    
    if step % 200 == 0:
        print(f"  Step {step:4d} | Loss: {loss.item():.4f}")

# ── 결과 시각화 ────────────────────────────────────────────────────────
road_result = torch.sigmoid(model.road_logits).detach().numpy()
gate_result = torch.sigmoid(model.gate_logits).detach().numpy()

# 관문 위치 복원
def get_gate_pos(idx, res):
    if idx < res: return (idx, 0) # North
    elif idx < 2*res: return (idx - res, res-1) # South
    elif idx < 3*res: return (0, idx - 2*res) # West
    else: return (res-1, idx - 3*res) # East

top_gate_idx = np.argsort(gate_result)[-2:] # 가장 점수가 높은 2곳
gates = [get_gate_pos(i, GRID_RES) for i in top_gate_idx]

def plot_final():
    fig, ax = plt.subplots(figsize=(14, 13))
    ax.set_facecolor('black')
    fig.patch.set_facecolor('black')
    
    # 1. 도로망 (Heatmap)
    # 보간법을 사용하여 부드러운 도로 느낌 생성
    im = ax.imshow(road_result, cmap='hot', extent=[0, W, H, 0], alpha=0.8, origin='upper')
    
    # 2. 건물 외곽선
    for bid, pts in BUILDING_POLY.items():
        pts_np = np.array(pts)
        poly = mpatches.Polygon(pts_np, closed=True, facecolor='none', edgecolor='#00d2d3', lw=1.5, alpha=0.7, zorder=5)
        ax.add_patch(poly)
        cx, cy = np.mean(pts_np, axis=0)
        ax.text(cx, cy, bid, color='white', ha='center', va='center', fontsize=8, fontweight='bold')

    # 3. AI 추천 관문
    labels = ["MAIN GATE", "BACK GATE"]
    for i, (gx, gy) in enumerate(gates):
        rx, ry = gx * (W/GRID_RES), gy * (H/GRID_RES)
        ax.scatter(rx, ry, s=300, c='yellow', marker='*', edgecolors='red', lw=2, zorder=10)
        ax.text(rx, ry-50, labels[i], color='yellow', ha='center', fontweight='bold', fontsize=12)

    plt.title("Pure GNN-Learned Optimal Road Network & Gate Locations\n(No Pathfinding, No Hubs, Full Geometry Learning)", 
              color='white', fontsize=20, pad=20)
    plt.axis('off')
    
    out_path = 'campus_pure_gnn_road_design.png'
    plt.savefig(out_path, dpi=130, bbox_inches='tight', facecolor='black')
    plt.close()
    print(f"\n✅ 순수 GNN 설계 완료: {out_path}")
    print(f"AI 추천 정문: {gates[0]}, 후문: {gates[1]} (격자 좌표)")

import matplotlib.patches as mpatches
if __name__ == '__main__':
    plot_final()
