"""
evaluate_road_design.py
이미 생성된 결과 이미지(campus_v16_civil_design.png / campus_pure_gnn_road_design.png)를
픽셀 단위로 분석해서 발표 슬라이드용 자체 정의 지표(충돌률 / 연결률 / 평균 차수)를 계산한다.

torch 재학습 없이, "최종 산출물(이미지)"을 근거로 측정 → 자체 정의 metric의 측정 방법 자체를 명시한다.

판별 기준:
- 도로(road) 픽셀  : hot 계열 colormap의 따뜻한 색 (R 高, B 低) — 두 이미지 공통으로 성립
- 건물(building) 윤곽 픽셀 : cyan/teal 계열 (G,B 高, R 低)
- 건물 영역(zone)  : 윤곽 픽셀을 N px 만큼 팽창(dilate)하여 근사 (폴리곤 내부를 채우는 대신,
  "도로가 건물 경계에 닿거나 겹치는지"를 충돌/연결의 판정 기준으로 사용)
"""
import numpy as np
from PIL import Image
import sys

DILATE_PX = 6          # 건물 윤곽선을 두껍게 만들어 "건물 영역"으로 근사할 반경
CONNECT_BUFFER_PX = 18  # 건물 군집 주변 이 거리 안에 도로 픽셀이 있으면 "연결됨"으로 판정
DOWNSCALE = 360         # 연결-요소(건물 개수) 라벨링을 위해 축소할 한 변의 길이


def load_masks(path):
    img = Image.open(path).convert('RGB')
    arr = np.asarray(img).astype(int)
    r, g, b = arr[..., 0], arr[..., 1], arr[..., 2]
    road_mask = (r > 140) & (r > b + 40)               # 노랑/빨강 계열 (hot colormap 상단부)
    building_mask = (g > 90) & (b > 90) & (r < g - 30)  # 청록색 윤곽선
    return road_mask, building_mask


def dilate(mask, radius):
    """단순 사각 커널 팽창 (scipy 없이 numpy 누적합으로 구현)."""
    if radius <= 0:
        return mask
    h, w = mask.shape
    pad = radius
    padded = np.zeros((h + 2 * pad, w + 2 * pad), dtype=np.int32)
    padded[pad:pad + h, pad:pad + w] = mask.astype(np.int32)
    csum = padded.cumsum(0).cumsum(1)
    csum = np.pad(csum, ((1, 0), (1, 0)))
    size = 2 * radius + 1
    out = np.zeros_like(mask, dtype=np.int32)
    for i in range(h):
        y0, y1 = i, i + size
        rowsum = csum[y1, :] - csum[y0, :]
        out[i, :] = (rowsum[size:] - rowsum[:-size]) if w >= size else 0
    # 위 1D 누적합 트릭은 행 방향만 처리되므로, 2D는 just brute dilation via shifting
    return _dilate_brute(mask, radius)


def _dilate_brute(mask, radius):
    out = mask.copy()
    h, w = mask.shape
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            if dx * dx + dy * dy > radius * radius:
                continue
            shifted = np.zeros_like(mask)
            ys, ye = max(0, dy), h + min(0, dy)
            xs, xe = max(0, dx), w + min(0, dx)
            ys2, ye2 = max(0, -dy), h + min(0, -dy)
            xs2, xe2 = max(0, -dx), w + min(0, -dx)
            shifted[ys:ye, xs:xe] = mask[ys2:ye2, xs2:xe2]
            out |= shifted
    return out


def avg_degree(road_mask):
    """8-방향 이웃 중 도로인 픽셀 수의 평균 (래스터 그래프에서의 평균 차수 근사)."""
    h, w = road_mask.shape
    m = road_mask.astype(np.int32)
    neighbor_sum = np.zeros_like(m)
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            if dy == 0 and dx == 0:
                continue
            shifted = np.zeros_like(m)
            ys, ye = max(0, dy), h + min(0, dy)
            xs, xe = max(0, dx), w + min(0, dx)
            ys2, ye2 = max(0, -dy), h + min(0, -dy)
            xs2, xe2 = max(0, -dx), w + min(0, -dx)
            shifted[ys:ye, xs:xe] = m[ys2:ye2, xs2:xe2]
            neighbor_sum += shifted
    road_pixels = m.astype(bool)
    if road_pixels.sum() == 0:
        return 0.0
    return float(neighbor_sum[road_pixels].mean())


def label_components(mask):
    """단순 BFS 기반 connected-component 라벨링 (scipy.ndimage.label 대체)."""
    h, w = mask.shape
    labels = np.zeros((h, w), dtype=np.int32)
    cur = 0
    visited = np.zeros((h, w), dtype=bool)
    ys, xs = np.where(mask)
    coords = set(zip(ys.tolist(), xs.tolist()))
    for sy, sx in zip(ys.tolist(), xs.tolist()):
        if visited[sy, sx]:
            continue
        cur += 1
        stack = [(sy, sx)]
        visited[sy, sx] = True
        size = 0
        while stack:
            y, x = stack.pop()
            labels[y, x] = cur
            size += 1
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    ny, nx = y + dy, x + dx
                    if 0 <= ny < h and 0 <= nx < w and not visited[ny, nx] and mask[ny, nx]:
                        visited[ny, nx] = True
                        stack.append((ny, nx))
    return labels, cur


def downscale_mask(mask, target):
    h, w = mask.shape
    fy, fx = max(1, h // target), max(1, w // target)
    small = mask[:fy * (h // fy), :fx * (w // fx)]
    small = small.reshape(h // fy, fy, w // fx, fx).any(axis=(1, 3))
    return small, fy, fx


def evaluate(path, label):
    print(f"\n===== {label} : {path} =====")
    road_mask, building_mask = load_masks(path)
    total_road = int(road_mask.sum())
    print(f"road pixel count     : {total_road}")
    print(f"building-outline px  : {int(building_mask.sum())}")

    # 1) 충돌률: 도로 픽셀 중 건물(팽창된) 영역과 겹치는 비율
    building_zone = dilate(building_mask, DILATE_PX)
    collision_px = int((road_mask & building_zone).sum())
    collision_rate = collision_px / total_road * 100 if total_road else 0.0
    print(f"[충돌률] {collision_rate:.2f}%  ({collision_px} / {total_road} px, dilate={DILATE_PX}px)")

    # 2) 평균 차수: 도로 픽셀의 8-이웃 중 도로인 픽셀 수 평균
    deg = avg_degree(road_mask)
    print(f"[평균 차수] {deg:.2f}  (8-이웃 중 도로 픽셀 평균 개수)")

    # 3) 연결률: 건물 군집(connected component) 수 대비, 인근에 도로가 있는 군집 수
    small_building, fy, fx = downscale_mask(building_mask, DOWNSCALE)
    labels, n_comp = label_components(small_building)
    small_road, _, _ = downscale_mask(road_mask, DOWNSCALE)
    buffer_r = max(1, CONNECT_BUFFER_PX // max(fy, fx))
    road_zone = _dilate_brute(small_road, buffer_r)
    connected = 0
    sizes = []
    for i in range(1, n_comp + 1):
        comp_mask = labels == i
        sizes.append(int(comp_mask.sum()))
        if (comp_mask & road_zone).any():
            connected += 1
    # 너무 작은 잡음 군집(라벨링 부산물) 제외하고 다시 집계
    sizes_arr = np.array(sizes)
    real = sizes_arr >= max(3, int(np.median(sizes_arr) * 0.25))
    n_real = int(real.sum())
    connected_real = sum(1 for i, ok in enumerate(real, start=0) if ok and ((labels == (i + 1)) & road_zone).any())
    print(f"[연결률(원시 라벨)] {connected}/{n_comp}  (군집 크기 분포 중앙값={int(np.median(sizes_arr))})")
    print(f"[연결률(잡음 제거)] {connected_real}/{n_real}  ≈ 실제 23개 건물 군집에 대응")

    return {
        'collision_rate': collision_rate,
        'avg_degree': deg,
        'connectivity_raw': (connected, n_comp),
        'connectivity_filtered': (connected_real, n_real),
    }


if __name__ == '__main__':
    results = {}
    results['V16'] = evaluate('campus_v16_civil_design.png', 'V16 (최종 모델)')
    results['baseline'] = evaluate('campus_pure_gnn_road_design.png', 'Baseline (초기 순수 GNN)')

    print("\n\n========== 요약 (발표 슬라이드용) ==========")
    for k, v in results.items():
        print(f"\n[{k}]")
        print(f"  충돌률     : {v['collision_rate']:.2f}%")
        print(f"  평균 차수  : {v['avg_degree']:.2f}")
        print(f"  연결률(원시): {v['connectivity_raw'][0]}/{v['connectivity_raw'][1]}")
        print(f"  연결률(필터): {v['connectivity_filtered'][0]}/{v['connectivity_filtered'][1]}")
