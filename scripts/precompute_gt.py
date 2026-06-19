"""
precompute_gt.py — OSM road_mask 기반 GT 사전 계산

OSM road_mask가 있는 캠퍼스(49개)에 대해 4개 알고리즘을 돌려
GT 픽셀 경로를 100x100 bool mask로 저장합니다.

출력: collegemap/gt_masks/{slug}_gt.npz
      keys: algo1, algo2, algo3, algo4  (각각 100x100 bool ndarray)

실행: python precompute_gt.py
      python precompute_gt.py --force   # 이미 있어도 재계산
"""

import os
import sys
import glob
import heapq
import argparse
import numpy as np
from PIL import Image
from scipy.ndimage import distance_transform_edt, maximum_filter, gaussian_filter
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra as sp_dijkstra

# ── 설정 ──────────────────────────────────────────────────────────────────────
RES             = 100
N               = RES * RES
CLUSTER_EPS     = 13.0
ROAD_CLEAR_MIN  = 3.0
ROAD_CLEAR_MAX  = 24.0
RIDGE_FILTER_SZ = 7
N_GATES         = 2
GATE_MIN_DIST   = 14

BASE_DIR  = os.path.dirname(os.path.abspath(__file__))
IMG_DIR   = os.path.join(BASE_DIR, "collegemap", "images")
TXT_DIR   = os.path.join(BASE_DIR, "collegemap", "txt")
ROAD_DIR  = os.path.join(BASE_DIR, "collegemap", "road_masks")
GT_DIR    = os.path.join(BASE_DIR, "collegemap", "gt_masks")
os.makedirs(GT_DIR, exist_ok=True)

# ── 헬퍼 (v28 load_campus와 동일) ─────────────────────────────────────────────
def _nearest_free_node(cy, cx, is_bld):
    yx = np.argwhere(is_bld == 0)
    if len(yx) == 0:
        return int(cy) * RES + int(cx)
    d = (yx[:, 0] - cy) ** 2 + (yx[:, 1] - cx) ** 2
    b = yx[np.argmin(d)]
    return int(b[0]) * RES + int(b[1])

def _make_ridge(is_bld):
    free  = is_bld == 0
    dist  = distance_transform_edt(free)
    clear = (dist > ROAD_CLEAR_MIN) & (dist < ROAD_CLEAR_MAX)
    inf   = gaussian_filter(is_bld.astype(np.float32), sigma=7.0) > 0.01
    lmax  = dist == maximum_filter(dist, size=RIDGE_FILTER_SZ)
    ridge = free & clear & inf & lmax
    rs    = gaussian_filter(ridge.astype(np.float32), sigma=1.2)
    if rs.max() < 1e-6:
        rs = gaussian_filter((free & clear & inf).astype(np.float32), sigma=1.0)
    return (rs / (rs.max() + 1e-6)).astype(np.float32)

def _cluster_centers(centers, eps=CLUSTER_EPS):
    n = len(centers); visited = [False] * n; clusters = []
    for i in range(n):
        if visited[i]: continue
        stack = [i]; visited[i] = True; cluster = []
        while stack:
            u = stack.pop(); cluster.append(u); uy, ux = centers[u]
            for v in range(n):
                if visited[v]: continue
                vy, vx = centers[v]
                if ((uy - vy) ** 2 + (ux - vx) ** 2) ** 0.5 <= eps:
                    visited[v] = True; stack.append(v)
        clusters.append(cluster)
    return clusters

def _nearest_ridge_node(cy, cx, is_bld, ridge):
    cands = np.argwhere((is_bld == 0) & (ridge > 0.05))
    if len(cands) == 0:
        return _nearest_free_node(cy, cx, is_bld)
    d    = (cands[:, 0] - cy) ** 2 + (cands[:, 1] - cx) ** 2
    best = cands[np.argmin(d)]
    return int(best[0]) * RES + int(best[1])

_BOUNDARY = sorted(set(
    list(range(10, RES - 10)) +
    [(RES - 1) * RES + i for i in range(10, RES - 10)] +
    [i * RES for i in range(10, RES - 10)] +
    [i * RES + (RES - 1) for i in range(10, RES - 10)]
))

def _select_gate_nodes(cluster_nodes, is_bld, ridge):
    if not cluster_nodes:
        return []
    cy = np.mean([n // RES for n in cluster_nodes])
    cx = np.mean([n % RES  for n in cluster_nodes])
    cands = []
    for node in _BOUNDARY:
        y, x = divmod(node, RES)
        if is_bld[y, x] > 0:
            continue
        score = ((y - cy) ** 2 + (x - cx) ** 2) ** 0.5 - ridge[y, x] * 8.0
        cands.append((score, node))
    cands.sort()
    chosen = []
    for _, node in cands:
        y, x = divmod(node, RES)
        if all(abs(x - (p % RES)) + abs(y - (p // RES)) >= GATE_MIN_DIST for p in chosen):
            chosen.append(node)
        if len(chosen) == N_GATES:
            break
    return chosen

# ── 그래프 / 경로 탐색 ─────────────────────────────────────────────────────────
def _make_graph(cost_flat):
    yc = np.repeat(np.arange(RES), RES)
    xc = np.tile(np.arange(RES), RES)
    rows, cols, vals = [], [], []
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            if dy == dx == 0:
                continue
            ny, nx = yc + dy, xc + dx
            ok = (ny >= 0) & (ny < RES) & (nx >= 0) & (nx < RES)
            u  = yc[ok] * RES + xc[ok]
            v  = ny[ok] * RES + nx[ok]
            w  = cost_flat[v] * (1.4142 if abs(dy) + abs(dx) == 2 else 1.0)
            rows.append(u); cols.append(v); vals.append(w)
    return csr_matrix(
        (np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
        shape=(N, N)
    )

def _dijkstra_mst(terminals, graph):
    n_t = len(terminals)
    if n_t <= 1 or n_t > 14:
        return set(terminals[:1] if terminals else [])
    idx  = np.array(terminals, dtype=np.int32)
    dist, prev = sp_dijkstra(graph, directed=True, indices=idx, return_predecessors=True)
    in_mst = [False] * n_t; in_mst[0] = True; mst_edges = []
    for _ in range(n_t - 1):
        bc, bi, bj = np.inf, -1, -1
        for i in range(n_t):
            if not in_mst[i]: continue
            for j in range(n_t):
                if in_mst[j]: continue
                d = dist[i, terminals[j]]
                if d < bc: bc, bi, bj = d, i, j
        if bi == -1 or not np.isfinite(bc): break
        in_mst[bj] = True; mst_edges.append((bi, bj))
    nodes = set()
    for i, j in mst_edges:
        cur = terminals[j]; src = terminals[i]; g = 0
        while cur != src and cur >= 0 and g < N:
            nodes.add(cur); cur = int(prev[i, cur]); g += 1
        nodes.add(src)
    return nodes

def _astar_path(src_n, dst_n, cost_flat, is_bld_flat):
    gy, gx = divmod(dst_n, RES)
    g_score = {src_n: 0.0}; came = {}
    h = lambda n: ((n // RES - gy) ** 2 + (n % RES - gx) ** 2) ** 0.5
    heap = [(h(src_n), 0.0, src_n)]
    while heap:
        _, gs, cur = heapq.heappop(heap)
        if cur == dst_n: break
        if gs > g_score.get(cur, np.inf): continue
        cy, cx = divmod(cur, RES)
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                if dy == dx == 0: continue
                ny, nx = cy + dy, cx + dx
                if not (0 <= ny < RES and 0 <= nx < RES): continue
                nb = ny * RES + nx
                if is_bld_flat[nb] > 0: continue
                step = cost_flat[nb] * (1.4142 if abs(dy) + abs(dx) == 2 else 1.0)
                ng   = gs + step
                if ng < g_score.get(nb, np.inf):
                    g_score[nb] = ng; came[nb] = cur
                    heapq.heappush(heap, (ng + h(nb), ng, nb))
    path = set(); cur = dst_n
    while cur in came: path.add(cur); cur = came[cur]
    path.add(src_n)
    return path

def _astar_mst(terminals, cost_flat, is_bld_flat):
    if len(terminals) <= 1:
        return set(terminals)
    if len(terminals) > 14:
        terminals = terminals[:14]
    in_mst = [False] * len(terminals); in_mst[0] = True
    connected = {terminals[0]}; nodes = {terminals[0]}
    for _ in range(len(terminals) - 1):
        best_path = None; best_cost = np.inf
        for j, t in enumerate(terminals):
            if in_mst[j]: continue
            for s in connected:
                p = _astar_path(s, t, cost_flat, is_bld_flat)
                c = sum(cost_flat[nn] for nn in p)
                if c < best_cost: best_cost = c; best_path = (p, j, t)
        if best_path is None: break
        p, j, t = best_path; nodes |= p; in_mst[j] = True; connected.add(t)
    return nodes

# ── 4알고리즘 GT 계산 ──────────────────────────────────────────────────────────
def compute_gt_variants(terminals, is_bld, road_map):
    """road_mask 기반 4종 GT 픽셀 집합 반환."""
    is_bld_flat = is_bld.flatten()
    road_flat   = road_map.flatten()

    # algo1: Dijkstra, cost = 1/(road+0.1)  — 등급 비례 선호
    c1 = np.where(road_flat > 0.4, 1.0 / (road_flat + 0.1), 50.0).astype(np.float32)
    c1[is_bld_flat > 0] = 1e6

    # algo2: Dijkstra, cost = uniform(1)    — 모든 도로 동등
    c2 = np.where(road_flat > 0.4, 1.0, 50.0).astype(np.float32)
    c2[is_bld_flat > 0] = 1e6

    # algo3: Dijkstra, cost = 1/(road²+0.01) — 높은 등급만 강하게 선호
    c3 = np.where(road_flat > 0.4, 1.0 / (road_flat ** 2 + 0.01), 50.0).astype(np.float32)
    c3[is_bld_flat > 0] = 1e6

    nodes1 = _dijkstra_mst(terminals, _make_graph(c1))
    nodes2 = _dijkstra_mst(terminals, _make_graph(c2))
    nodes3 = _dijkstra_mst(terminals, _make_graph(c3))

    # algo4: A*, cost = c1 + 직선 편향
    nodes4 = _astar_mst(terminals, c1, is_bld_flat)

    results = {}
    for name, node_set in [('algo1', nodes1), ('algo2', nodes2),
                            ('algo3', nodes3), ('algo4', nodes4)]:
        mask = np.zeros(N, dtype=bool)
        for px in node_set:
            if 0 <= px < N:
                mask[px] = True
        results[name] = mask.reshape(RES, RES)

    return results

# ── 캠퍼스 처리 ────────────────────────────────────────────────────────────────
def _find_txt(slug):
    p = os.path.join(TXT_DIR, f"{slug}_building_places.txt")
    if os.path.exists(p): return p
    prefix = slug.split('_')[0]
    for fn in os.listdir(TXT_DIR):
        if fn.endswith('_building_places.txt') and fn.startswith(prefix):
            return os.path.join(TXT_DIR, fn)
    return None

def process(slug, force=False):
    out_path = os.path.join(GT_DIR, f"{slug}_gt.npz")
    if os.path.exists(out_path) and not force:
        print(f"  [SKIP] {slug}")
        return True

    img_path  = os.path.join(IMG_DIR,  f"{slug}_building_mask.png")
    road_path = os.path.join(ROAD_DIR, f"{slug}_road_mask.npy")
    txt_path  = _find_txt(slug)

    if not all(os.path.exists(p) for p in [img_path, road_path] if p) or txt_path is None:
        print(f"  [SKIP] 파일 없음: {slug}")
        return False

    print(f"  ▶ {slug}", end="", flush=True)

    # 빌딩 마스크
    img    = Image.open(img_path).convert('L')
    W, H   = img.size
    is_bld = (np.array(img.resize((RES, RES), resample=Image.NEAREST)) > 128).astype(np.float32)

    # ridge (터미널 선택용)
    ridge = _make_ridge(is_bld)

    # road_mask
    road_map = np.load(road_path)   # 이미 100x100

    # 건물 폴리곤 → 클러스터 → 터미널
    ns = {}; exec(open(txt_path, encoding='utf-8').read(), ns)
    poly = ns['BUILDING_POLY']; building_ids = list(poly.keys())
    centers = []
    for bid in building_ids:
        pts = poly[bid]
        cy  = np.mean([p[1] for p in pts]) * RES / H
        cx  = np.mean([p[0] for p in pts]) * RES / W
        centers.append((cy, cx))

    bld_nodes = [_nearest_free_node(cy, cx, is_bld) for cy, cx in centers]
    clusters  = _cluster_centers(centers)
    cluster_nodes = []
    for cl in clusters:
        cy   = np.mean([centers[i][0] for i in cl])
        cx   = np.mean([centers[i][1] for i in cl])
        node = _nearest_ridge_node(cy, cx, is_bld, ridge)
        if node not in cluster_nodes:
            cluster_nodes.append(node)
    if len(cluster_nodes) <= 1 and len(bld_nodes) > 1:
        cluster_nodes = list(dict.fromkeys(bld_nodes))

    gate_nodes = _select_gate_nodes(cluster_nodes, is_bld, ridge)
    terminals  = list(dict.fromkeys(gate_nodes + cluster_nodes))

    # 4알고리즘 GT 계산
    variants = compute_gt_variants(terminals, is_bld, road_map)

    # 저장
    np.savez_compressed(out_path, **variants)

    total_px = {k: int(v.sum()) for k, v in variants.items()}
    print(f"  →  {total_px}")
    return True


# ── 메인 ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--force', action='store_true', help='이미 있는 파일도 재계산')
    args = parser.parse_args()

    road_mask_paths = sorted(glob.glob(os.path.join(ROAD_DIR, '*_road_mask.npy')))
    slugs = [os.path.basename(p).replace('_road_mask.npy', '') for p in road_mask_paths]

    print(f"총 {len(slugs)}개 캠퍼스 GT 계산 시작\n")
    ok, fail = 0, []
    for i, slug in enumerate(slugs, 1):
        print(f"[{i:2d}/{len(slugs)}]", end=" ")
        try:
            if process(slug, force=args.force):
                ok += 1
            else:
                fail.append(slug)
        except Exception as e:
            print(f"  [ERROR] {e}")
            fail.append(slug)

    print(f"\n완료: 성공 {ok}개 / 실패 {len(fail)}개")
    if fail:
        print("실패:", fail)
