"""
precompute_gt_betweenness.py — terminal 간 betweenness centrality로 필수 도로 탐지

아이디어:
  4방향 gate + cluster_nodes를 terminal로 잡고,
  모든 terminal 쌍(i,j) 최단 경로를 구해 픽셀별 경유 횟수를 셈.
  많은 경로가 지나간 픽셀 = 어떤 경로든 반드시 통과하는 병목 = 필수 도로.

  cost function 4종 각각 betweenness를 구해 합산.

출력: collegemap/gt_masks_betweenness/{slug}_gt.npz
  vote      : 픽셀별 경유 횟수 합계 (4 cost × 모든 쌍)
  vote_norm : vote / vote.max() (float32, 0~1)
  b10, b25, b50, b75 : vote_norm >= 0.10 / 0.25 / 0.50 / 0.75
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
GT_DIR   = os.path.join(BASE_DIR, "collegemap", "gt_masks_betweenness")
os.makedirs(GT_DIR, exist_ok=True)

# ── 헬퍼 ─────────────────────────────────────────────────────────────────────
def _nearest_free_node(cy, cx, is_bld):
    yx = np.argwhere(is_bld == 0)
    if len(yx) == 0: return int(cy)*RES + int(cx)
    d = (yx[:,0]-cy)**2 + (yx[:,1]-cx)**2
    b = yx[np.argmin(d)]
    return int(b[0])*RES + int(b[1])

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
    return int(cands[np.argmin(d)][0])*RES + int(cands[np.argmin(d)][1])

def _select_gates_4dir(is_bld, ridge):
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
            if node not in gates: gates.append(node)
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

# ── Betweenness ───────────────────────────────────────────────────────────────
def _betweenness_vote(terminals, graph):
    """모든 terminal 쌍 최단 경로 → 픽셀별 경유 횟수."""
    k = len(terminals)
    if k < 2: return np.zeros(N, dtype=np.int32)

    idx = np.array(terminals, dtype=np.int32)
    _, prev = sp_dijkstra(graph, directed=True, indices=idx,
                          return_predecessors=True)
    vote = np.zeros(N, dtype=np.int32)
    for i in range(k):
        for j in range(k):
            if i == j: continue
            cur = terminals[j]; g = 0
            while cur != terminals[i] and cur >= 0 and g < N:
                vote[cur] += 1
                cur = int(prev[i, cur]); g += 1
            vote[terminals[i]] += 1
    return vote

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

def _astar_betweenness_vote(terminals, cost_flat, is_bld_flat):
    """모든 terminal 쌍 A* 경로 → 픽셀별 경유 횟수."""
    k = len(terminals)
    vote = np.zeros(N, dtype=np.int32)
    for i in range(k):
        for j in range(i+1, k):   # undirected: (i,j)와 (j,i) 경로 동일
            path = _astar_path(terminals[i], terminals[j], cost_flat, is_bld_flat)
            for px in path:
                if 0 <= px < N: vote[px] += 2  # 양방향 count
    return vote

# ── 메인 계산 ─────────────────────────────────────────────────────────────────
def compute_betweenness(terminals, is_bld, road_map):
    is_bld_flat = is_bld.flatten()
    road_flat   = road_map.flatten()
    road_norm   = road_flat / (road_flat.max() + 1e-6)

    c1 = np.where(road_norm>0.1, 1.0/(road_norm+0.1),    50.0).astype(np.float32)
    c2 = np.where(road_norm>0.1, 1.0,                     50.0).astype(np.float32)
    c3 = np.where(road_norm>0.1, 1.0/(road_norm**2+0.01), 50.0).astype(np.float32)
    for c in (c1, c2, c3): c[is_bld_flat>0] = 1e6

    vote = np.zeros(N, dtype=np.int32)
    vote += _betweenness_vote(terminals, _make_graph(c1))
    vote += _betweenness_vote(terminals, _make_graph(c2))
    vote += _betweenness_vote(terminals, _make_graph(c3))
    vote += _astar_betweenness_vote(terminals, c1, is_bld_flat)

    vote_2d = vote.reshape(RES, RES)
    mx = vote_2d.max()
    vote_norm = (vote_2d / (mx + 1e-6)).astype(np.float32)

    return {
        'vote':      vote_2d,
        'vote_norm': vote_norm,
        'b10':  (vote_norm >= 0.10),
        'b25':  (vote_norm >= 0.25),
        'b50':  (vote_norm >= 0.50),
        'b75':  (vote_norm >= 0.75),
    }

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
    terminals  = list(dict.fromkeys(gate_nodes + cluster_nodes))
    if len(terminals) > 20:
        terminals = terminals[:20]  # gate 우선, 최대 20개 (pairs 최대 380)

    k = len(terminals)
    n_pairs = k*(k-1)
    print(f"  terminals={k}  pairs={n_pairs}", end="", flush=True)

    result = compute_betweenness(terminals, is_bld, road_map)
    np.savez_compressed(out_path, **result)

    print(f"  b25={int(result['b25'].sum())}px  b50={int(result['b50'].sum())}px  b75={int(result['b75'].sum())}px")
    return True

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--force', action='store_true')
    args = parser.parse_args()

    slugs = [os.path.basename(p).replace('_road_mask.npy','')
             for p in sorted(glob.glob(os.path.join(ROAD_DIR,'*_road_mask.npy')))]

    print(f"총 {len(slugs)}개  (betweenness: 4 cost × 모든 terminal 쌍)\n")
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
