"""
precompute_gt_4gate.py — 상하좌우 4방향 gate × 4 cost = 16 variant 투표로 consensus GT

아이디어:
  4방향(상/하/좌/우)에서 각각 출발해 cluster_nodes까지 경로를 찾으면,
  여러 방향이 공통으로 지나는 픽셀 = 어디서 오든 반드시 통과해야 하는 도로.

cost functions (기존 precompute_gt.py와 동일):
  c1: Dijkstra, 1/(road+0.1)
  c2: Dijkstra, uniform(1)
  c3: Dijkstra, 1/(road²+0.01)
  c4: A* (c1 cost + 직선 편향)

출력: collegemap/gt_masks_4gate/{slug}_gt.npz
      keys:
        v{i}        (i=0..15) — 16개 개별 variant (100x100 bool)
        vote        — 픽셀별 동의 수 (0~16, int8)
        consensus_4 — vote >= 4
        consensus_8 — vote >= 8
        consensus_12— vote >= 12
"""

import os, glob, heapq, argparse
import numpy as np
from PIL import Image
from scipy.ndimage import distance_transform_edt, maximum_filter, gaussian_filter
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra as sp_dijkstra

RES            = 100
N              = RES * RES
CLUSTER_EPS    = 13.0
ROAD_CLEAR_MIN = 3.0
ROAD_CLEAR_MAX = 24.0
RIDGE_FILTER_SZ= 7

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
IMG_DIR  = os.path.join(BASE_DIR, "collegemap", "images")
TXT_DIR  = os.path.join(BASE_DIR, "collegemap", "txt")
ROAD_DIR = os.path.join(BASE_DIR, "collegemap", "road_masks")
GT_DIR   = os.path.join(BASE_DIR, "collegemap", "gt_masks_4gate")
os.makedirs(GT_DIR, exist_ok=True)

# ── 헬퍼 ─────────────────────────────────────────────────────────────────────
def _nearest_free_node(cy, cx, is_bld):
    yx = np.argwhere(is_bld == 0)
    if len(yx) == 0: return int(cy)*RES + int(cx)
    d = (yx[:,0]-cy)**2 + (yx[:,1]-cx)**2
    return int(yx[np.argmin(d)][0])*RES + int(yx[np.argmin(d)][1])

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
    n = len(centers); visited = [False]*n; clusters = []
    for i in range(n):
        if visited[i]: continue
        stack=[i]; visited[i]=True; cluster=[]
        while stack:
            u=stack.pop(); cluster.append(u); uy,ux=centers[u]
            for v in range(n):
                if visited[v]: continue
                vy,vx=centers[v]
                if ((uy-vy)**2+(ux-vx)**2)**0.5 <= eps:
                    visited[v]=True; stack.append(v)
        clusters.append(cluster)
    return clusters

def _nearest_ridge_node(cy, cx, is_bld, ridge):
    cands = np.argwhere((is_bld==0)&(ridge>0.05))
    if len(cands)==0: return _nearest_free_node(cy,cx,is_bld)
    d = (cands[:,0]-cy)**2+(cands[:,1]-cx)**2
    best = cands[np.argmin(d)]
    return int(best[0])*RES+int(best[1])

def _select_gates_4dir(is_bld, ridge):
    """상/하/좌/우 각 방향에서 ridge score 가장 높은 free pixel 1개씩."""
    margin = 8
    sides = {
        'top':    [(r,c) for r in range(margin) for c in range(margin, RES-margin)],
        'bottom': [(r,c) for r in range(RES-margin, RES) for c in range(margin, RES-margin)],
        'left':   [(r,c) for r in range(margin, RES-margin) for c in range(margin)],
        'right':  [(r,c) for r in range(margin, RES-margin) for c in range(RES-margin, RES)],
    }
    gates = []
    for coords in sides.values():
        free = [(r,c) for r,c in coords if is_bld[r,c] == 0]
        if free:
            best = max(free, key=lambda rc: ridge[rc[0], rc[1]])
            node = best[0]*RES + best[1]
            if node not in gates:
                gates.append(node)
    return gates

def _make_graph(cost_flat):
    yc = np.repeat(np.arange(RES), RES)
    xc = np.tile(np.arange(RES), RES)
    rows, cols, vals = [], [], []
    for dy in (-1,0,1):
        for dx in (-1,0,1):
            if dy==dx==0: continue
            ny,nx = yc+dy, xc+dx
            ok = (ny>=0)&(ny<RES)&(nx>=0)&(nx<RES)
            u = yc[ok]*RES+xc[ok]; v = ny[ok]*RES+nx[ok]
            w = cost_flat[v]*(1.4142 if abs(dy)+abs(dx)==2 else 1.0)
            rows.append(u); cols.append(v); vals.append(w)
    return csr_matrix(
        (np.concatenate(vals),(np.concatenate(rows),np.concatenate(cols))),
        shape=(N,N))

def _dijkstra_paths(src_node, targets, graph):
    """src → 각 target 까지 최단 경로 픽셀 union."""
    _, prev = sp_dijkstra(graph, directed=True, indices=[src_node],
                          return_predecessors=True)
    nodes = {src_node}
    for t in targets:
        cur = t; g = 0
        while cur != src_node and cur >= 0 and g < N:
            nodes.add(cur); cur = int(prev[0, cur]); g += 1
        nodes.add(src_node)
    return nodes

def _astar_path(src_n, dst_n, cost_flat, is_bld_flat):
    gy, gx = divmod(dst_n, RES)
    g_score = {src_n: 0.0}; came = {}
    h = lambda n: ((n//RES-gy)**2+(n%RES-gx)**2)**0.5
    heap = [(h(src_n), 0.0, src_n)]
    while heap:
        _, gs, cur = heapq.heappop(heap)
        if cur == dst_n: break
        if gs > g_score.get(cur, np.inf): continue
        cy, cx = divmod(cur, RES)
        for dy in (-1,0,1):
            for dx in (-1,0,1):
                if dy==dx==0: continue
                ny,nx = cy+dy, cx+dx
                if not (0<=ny<RES and 0<=nx<RES): continue
                nb = ny*RES+nx
                if is_bld_flat[nb] > 0: continue
                step = cost_flat[nb]*(1.4142 if abs(dy)+abs(dx)==2 else 1.0)
                ng = gs+step
                if ng < g_score.get(nb, np.inf):
                    g_score[nb]=ng; came[nb]=cur
                    heapq.heappush(heap,(ng+h(nb),ng,nb))
    path=set(); cur=dst_n
    while cur in came: path.add(cur); cur=came[cur]
    path.add(src_n)
    return path

def _astar_paths(src_node, targets, cost_flat, is_bld_flat):
    nodes = {src_node}
    for t in targets:
        nodes |= _astar_path(src_node, t, cost_flat, is_bld_flat)
    return nodes

def compute_gt_4gate(gate_nodes, cluster_nodes, is_bld, road_map):
    is_bld_flat = is_bld.flatten()
    road_flat   = road_map.flatten()
    road_norm   = road_flat / (road_flat.max() + 1e-6)

    c1 = np.where(road_norm>0.1, 1.0/(road_norm+0.1),       50.0).astype(np.float32)
    c2 = np.where(road_norm>0.1, 1.0,                         50.0).astype(np.float32)
    c3 = np.where(road_norm>0.1, 1.0/(road_norm**2+0.01),    50.0).astype(np.float32)
    for c in (c1, c2, c3): c[is_bld_flat>0] = 1e6
    c1_astar = c1.copy()  # A*도 c1 cost 사용

    g1, g2, g3 = _make_graph(c1), _make_graph(c2), _make_graph(c3)
    targets = cluster_nodes if cluster_nodes else gate_nodes

    variants = []
    for gate in gate_nodes:
        for graph, c in [(g1,None),(g2,None),(g3,None)]:
            node_set = _dijkstra_paths(gate, targets, graph)
            mask = np.zeros(N, dtype=bool)
            for px in node_set:
                if 0 <= px < N: mask[px] = True
            variants.append(mask.reshape(RES,RES))
        # A*
        node_set = _astar_paths(gate, targets, c1_astar, is_bld_flat)
        mask = np.zeros(N, dtype=bool)
        for px in node_set:
            if 0 <= px < N: mask[px] = True
        variants.append(mask.reshape(RES,RES))

    vote = np.zeros((RES,RES), dtype=np.int8)
    for v in variants:
        vote += v.astype(np.int8)

    result = {f'v{i}': variants[i] for i in range(len(variants))}
    result['vote']         = vote
    result['consensus_4']  = (vote >= 4)   # 25%
    result['consensus_8']  = (vote >= 8)   # 50%
    result['consensus_12'] = (vote >= 12)  # 75%
    result['consensus_16'] = (vote >= 16)  # 100% — 모든 알고리즘 동의
    return result

# ── 캠퍼스 처리 ──────────────────────────────────────────────────────────────
def _find_txt(slug):
    p = os.path.join(TXT_DIR, f"{slug}_building_places.txt")
    if os.path.exists(p): return p
    for fn in os.listdir(TXT_DIR):
        if fn.endswith('_building_places.txt') and fn.startswith(slug.split('_')[0]):
            return os.path.join(TXT_DIR, fn)
    return None

def process(slug, force=False):
    out_path = os.path.join(GT_DIR, f"{slug}_gt.npz")
    if os.path.exists(out_path) and not force:
        print(f"  [SKIP] {slug}"); return True

    img_path  = os.path.join(IMG_DIR,  f"{slug}_building_mask.png")
    road_path = os.path.join(ROAD_DIR, f"{slug}_road_mask.npy")
    txt_path  = _find_txt(slug)
    if not os.path.exists(img_path) or not os.path.exists(road_path) or txt_path is None:
        print(f"  [SKIP] 파일없음: {slug}"); return False

    print(f"  ▶ {slug}", end="", flush=True)

    img    = Image.open(img_path).convert('L')
    W, H   = img.size
    is_bld = (np.array(img.resize((RES,RES), Image.NEAREST))>128).astype(np.float32)
    ridge  = _make_ridge(is_bld)
    road_map = np.load(road_path)

    ns = {}; exec(open(txt_path, encoding='utf-8').read(), ns)
    poly = ns['BUILDING_POLY']
    centers = []
    for bid, pts in poly.items():
        cy = np.mean([p[1] for p in pts])*RES/H
        cx = np.mean([p[0] for p in pts])*RES/W
        centers.append((cy,cx))

    bld_nodes = [_nearest_free_node(cy,cx,is_bld) for cy,cx in centers]
    clusters  = _cluster_centers(centers)
    cluster_nodes = []
    for cl in clusters:
        cy = np.mean([centers[i][0] for i in cl])
        cx = np.mean([centers[i][1] for i in cl])
        node = _nearest_ridge_node(cy,cx,is_bld,ridge)
        if node not in cluster_nodes: cluster_nodes.append(node)
    if len(cluster_nodes)<=1 and len(bld_nodes)>1:
        cluster_nodes = list(dict.fromkeys(bld_nodes))

    gate_nodes = _select_gates_4dir(is_bld, ridge)
    if not gate_nodes:
        print("  [SKIP] gate 없음"); return False

    result = compute_gt_4gate(gate_nodes, cluster_nodes, is_bld, road_map)
    np.savez_compressed(out_path, **result)

    n_vars = len([k for k in result if k.startswith('v')])
    print(f"  gate={len(gate_nodes)}  vars={n_vars}  "
          f"c4={int(result['consensus_4'].sum())}px  "
          f"c8={int(result['consensus_8'].sum())}px  "
          f"c12={int(result['consensus_12'].sum())}px  "
          f"c16={int(result['consensus_16'].sum())}px")
    return True

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--force', action='store_true')
    args = parser.parse_args()

    slugs = [os.path.basename(p).replace('_road_mask.npy','')
             for p in sorted(glob.glob(os.path.join(ROAD_DIR,'*_road_mask.npy')))]

    print(f"총 {len(slugs)}개  (4방향 gate × 4 cost = 최대 16 variant)\n")
    ok, fail = 0, []
    for i, slug in enumerate(slugs, 1):
        print(f"[{i:2d}/{len(slugs)}]", end=" ")
        try:
            if process(slug, args.force): ok += 1
            else: fail.append(slug)
        except Exception as e:
            print(f"  [ERROR] {e}"); fail.append(slug)

    print(f"\n완료: {ok}개 / 실패 {len(fail)}개")
    if fail: print("실패:", fail)
