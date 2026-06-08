"""
campus_realistic_designer.py
건물 폴리곤을 '장애물'로 인식하고, 이를 피해서 가는 최단 경로망을 설계합니다.
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathfinding.core.diagonal_movement import DiagonalMovement
from pathfinding.core.grid import Grid
from pathfinding.finder.a_star import AStarFinder
from PIL import Image, ImageDraw
import torch
import torch.nn as nn
import os
import sys
from scipy.ndimage import distance_transform_edt

# 데이터 로드
sys.path.append(os.getcwd())
try:
    from building_places import BUILDING_POLY
except ImportError:
    with open('building_places.txt', 'r') as f:
        content = f.read()
        # "BUILDING_POLY = { ... }" 형태의 텍스트에서 딕셔너리만 추출
        exec(content, globals())

# ── 설정 ──────────────────────────────────────────────────────────────────
W, H = 2223, 2056
SCALE = 0.1 
sW, sH = int(W * SCALE), int(H * SCALE)
HUBS = ['03', '48']

# 1. 장애물 마스크 생성 (1: 통과가능, 0: 통과불가)
mask_img = Image.new('L', (W, H), 1)
draw = ImageDraw.Draw(mask_img)
for pts in BUILDING_POLY.values():
    draw.polygon(pts, fill=0)

mask_small = mask_img.resize((sW, sH), resample=Image.NEAREST)
mask_np = np.array(mask_small) # (sH, sW)

# 2. 각 건물별 '입구(Road Access Point)' 찾기
# 건물 중심점은 건물 내부(0)이므로, 건물 밖(1) 중 가장 가까운 점을 찾음
dist_to_free = distance_transform_edt(1 - mask_np) # 0(자유공간)으로부터의 거리

def get_access_point(pts):
    p = np.array(pts)
    center = np.mean(p, axis=0) * SCALE
    cx, cy = int(center[0]), int(center[1])
    cx = max(0, min(sW-1, cx))
    cy = max(0, min(sH-1, cy))
    
    # 건물 내부라면 가장 가까운 빈 공간(1)을 찾음
    if mask_np[cy, cx] == 0:
        # 건물 주변 20px 내에서 탐색
        y_min, y_max = max(0, cy-20), min(sH, cy+21)
        x_min, x_max = max(0, cx-20), min(sW, cx+21)
        sub_mask = mask_np[y_min:y_max, x_min:x_max]
        if np.any(sub_mask == 1):
            coords = np.argwhere(sub_mask == 1)
            dists = np.sum((coords - [cy-y_min, cx-x_min])**2, axis=1)
            best_idx = np.argmin(dists)
            ry, rx = coords[best_idx]
            return rx + x_min, ry + y_min
    return cx, cy

BUILDING_IDS = list(BUILDING_POLY.keys())
ACCESS_POINTS = {bid: get_access_point(pts) for bid, pts in BUILDING_POLY.items()}

# 3. 모든 건물 쌍 간의 '진짜' 거리 계산 (A*)
finder = AStarFinder(diagonal_movement=DiagonalMovement.always)

def get_real_dist(bid1, bid2):
    p1 = ACCESS_POINTS[bid1]
    p2 = ACCESS_POINTS[bid2]
    
    grid = Grid(matrix=mask_np.tolist())
    try:
        start = grid.node(p1[0], p1[1])
        end = grid.node(p2[0], p2[1])
        path, runs = finder.find_path(start, end, grid)
        if not path or len(path) == 0:
            return np.linalg.norm(np.array(p1) - np.array(p2)) * 2, []
        return len(path), path
    except:
        return np.linalg.norm(np.array(p1) - np.array(p2)) * 2, []

print("▶ 건물 간 현실적 거리 계산 중 (Pathfinding)...")
dist_matrix = np.zeros((len(BUILDING_IDS), len(BUILDING_IDS)))
paths_dict = {}

for i, b1 in enumerate(BUILDING_IDS):
    for j, b2 in enumerate(BUILDING_IDS):
        if i >= j: continue
        d, path = get_real_dist(b1, b2)
        dist_matrix[i, j] = dist_matrix[j, i] = d
        paths_dict[(i, j)] = path

# 4. GNN 기반 연결 최적화
edge_weights = nn.Parameter(torch.randn(len(BUILDING_IDS), len(BUILDING_IDS)) * 0.01)
optimizer = torch.optim.Adam([edge_weights], lr=0.1)

print("▶ AI 최적 도로망 선정 중...")
for step in range(501):
    optimizer.zero_grad()
    adj = torch.sigmoid((edge_weights + edge_weights.t()) / 2)
    
    # Loss: 건설 비용 + 이동 효율성
    dist_t = torch.tensor(dist_matrix, dtype=torch.float)
    cost = torch.sum(adj * dist_t)
    
    efficiency = 0
    for hub in HUBS:
        h_idx = BUILDING_IDS.index(hub)
        efficiency += torch.mean(1.0 / (adj[h_idx, :] + 1e-2))
    
    loss = 0.0005 * cost + 1.0 * efficiency + 0.1 * torch.norm(adj, p=1)
    loss.backward()
    optimizer.step()

# 5. 시각화
final_adj = torch.sigmoid((edge_weights + edge_weights.t()) / 2).detach().numpy()
# 엣지 선별: 가중치가 높은 순서대로 뽑되, 너무 많지 않게
flat_adj = final_adj[np.triu_indices(len(BUILDING_IDS), k=1)]
threshold = np.percentile(flat_adj, 92)

def plot_realistic_design():
    fig, ax = plt.subplots(figsize=(15, 14))
    ax.set_facecolor('#0d0d0d')
    fig.patch.set_facecolor('#0d0d0d')
    
    # 건물 폴리곤
    for bid, pts in BUILDING_POLY.items():
        pts_np = np.array(pts)
        is_hub = bid in HUBS
        color = '#ff5252' if is_hub else '#4b7bec'
        poly = mpatches.Polygon(pts_np, closed=True, facecolor=color, alpha=0.5, edgecolor='white', lw=0.8, zorder=4)
        ax.add_patch(poly)
        cx, cy = np.mean(pts_np, axis=0)
        ax.text(cx, cy, bid, color='white', ha='center', va='center', fontsize=9, fontweight='bold', zorder=5)

    # AI 추천 도로 (A* 경로)
    for (i, j), path in paths_dict.items():
        w = final_adj[i, j]
        if w > threshold and len(path) > 1:
            path_np = np.array([(p.x, p.y) for p in path]) / SCALE
            lw = (w - threshold) / (1.0 - threshold) * 7 + 1.5
            ax.plot(path_np[:, 0], path_np[:, 1], color='#00d2d3', lw=lw, alpha=0.8, zorder=2)
            
            # 입구 포인트 표시 (작게)
            ax.scatter(path_np[0,0], path_np[0,1], s=10, color='#00d2d3', alpha=0.5, zorder=3)

    plt.title("AI-Generated Optimal Campus Road Network\n[Buildings as Obstacles | Hubs: 03, 48]", color='white', fontsize=22, pad=20)
    plt.axis('off')
    ax.set_xlim(0, W)
    ax.set_ylim(H, 0)
    
    out_path = 'campus_realistic_road_design.png'
    plt.savefig(out_path, dpi=130, bbox_inches='tight', facecolor='#0d0d0d')
    plt.close()
    print(f"\n✅ 설계 도면 저장 완료: {out_path}")

if __name__ == '__main__':
    plot_realistic_design()
