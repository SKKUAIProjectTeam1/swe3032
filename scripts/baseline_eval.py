"""Ridge-only / MST 베이스라인 평가 (학습 불필요).

train_v28_subgraph_gnn.ipynb cell 3~6의 그래프 구성 코드를 재사용해
49개 OSM 캠퍼스(flip 증강 없는 원본)에 대해
Ridge-only(임계값 기반)와 MST(Steiner tree 근사) 베이스라인의
Precision/Recall/F1을 계산한다.

Precision/Recall은 V28 노트북과 동일하게 "캠퍼스별로 계산 후 평균"(macro)한다.
"""
import os, glob, random
import numpy as np
import torch
from PIL import Image
from scipy.ndimage import distance_transform_edt, maximum_filter, gaussian_filter
import networkx as nx
from networkx.algorithms.approximation import steiner_tree

RES = 100
N = RES * RES
DEVICE = torch.device('cpu')

CLUSTER_EPS          = 13.0
CLUSTER_SPLIT_RADIUS = 15.0
ROAD_CLEAR_MIN    = 3.0
ROAD_CLEAR_MAX    = 24.0
RIDGE_FILTER_SIZE = 7
EDGE_THR          = 0.50
N_GATES       = 2
GATE_MIN_DIST = 14

IMG_DIR      = 'collegemap/images'
TXT_DIR      = 'collegemap/txt'
GT_FINAL_DIR = 'collegemap/gt_masks_final'


# ── cell 4: 공통 헬퍼 (원본과 동일) ─────────────────────────────────────────
def _nearest_free_node(cy, cx, is_bld_grid):
    non_bld_yx = np.argwhere(is_bld_grid == 0)
    if len(non_bld_yx) == 0: return int(cy)*RES + int(cx)
    dists = (non_bld_yx[:,0]-cy)**2 + (non_bld_yx[:,1]-cx)**2
    best  = non_bld_yx[np.argmin(dists)]
    return int(best[0])*RES + int(best[1])

def _make_common_road_ridge(is_bld_grid):
    free  = is_bld_grid == 0
    dist  = distance_transform_edt(free)
    clear = (dist > ROAD_CLEAR_MIN) & (dist < ROAD_CLEAR_MAX)
    campus_influence = gaussian_filter(is_bld_grid.astype(np.float32), sigma=7.0)
    near_campus = campus_influence > 0.01
    local_max   = dist == maximum_filter(dist, size=RIDGE_FILTER_SIZE)
    ridge       = free & clear & near_campus & local_max
    ridge_score = gaussian_filter(ridge.astype(np.float32), sigma=1.2)
    if ridge_score.max() < 1e-6:
        band = free & clear & near_campus
        ridge_score = gaussian_filter(band.astype(np.float32), sigma=1.0)
    ridge_score = ridge_score / (ridge_score.max() + 1e-6)
    return ridge_score.astype(np.float32), dist.astype(np.float32)

def _cluster_centers(centers, eps=CLUSTER_EPS):
    n=len(centers); assigned=[False]*n; clusters=[]
    for i in range(n):
        if assigned[i]: continue
        cy,cx=centers[i]; assigned[i]=True; cluster=[i]
        for j in range(n):
            if not assigned[j]:
                vy,vx=centers[j]
                if ((vy-cy)**2+(vx-cx)**2)**0.5 <= eps:
                    assigned[j]=True; cluster.append(j)
        clusters.append(cluster)
    return clusters

def _nearest_ridge_node(cy, cx, is_bld_grid, ridge_grid):
    candidates = np.argwhere((is_bld_grid==0) & (ridge_grid>0.05))
    if len(candidates) == 0:
        return _nearest_free_node(cy, cx, is_bld_grid)
    dists = (candidates[:,0]-cy)**2 + (candidates[:,1]-cx)**2
    best  = candidates[np.argmin(dists)]
    return int(best[0])*RES + int(best[1])

_BOUNDARY = sorted(set(
    [i for i in range(10, RES-10)] +
    [(RES-1)*RES+i for i in range(10, RES-10)] +
    [i*RES for i in range(10, RES-10)] +
    [i*RES+(RES-1) for i in range(10, RES-10)]
))

def _select_gate_nodes(cluster_nodes, is_bld_grid, ridge_grid):
    if len(cluster_nodes) == 0: return []
    cy = np.mean([n//RES for n in cluster_nodes])
    cx = np.mean([n%RES  for n in cluster_nodes])
    candidates = []
    for node in _BOUNDARY:
        y, x = divmod(int(node), RES)
        if is_bld_grid[y, x] > 0: continue
        score = ((y-cy)**2+(x-cx)**2)**0.5 - ridge_grid[y,x]*8.0
        candidates.append((score, int(node)))
    candidates.sort(key=lambda t: t[0])
    chosen = []
    for _, node in candidates:
        y, x = divmod(node, RES)
        if all(abs(x-(p%RES))+abs(y-(p//RES)) >= GATE_MIN_DIST for p in chosen):
            chosen.append(node)
        if len(chosen) == N_GATES: break
    return chosen


# ── cell 5: 서브그래프 구성 (원본과 동일) ───────────────────────────────────
def build_subgraph(is_bld_grid):
    is_bld_flat = is_bld_grid.flatten()
    free_pixels = np.where(is_bld_flat == 0)[0].astype(np.int32)
    n_free = len(free_pixels)

    pixel_to_node = np.full(N, -1, dtype=np.int32)
    pixel_to_node[free_pixels] = np.arange(n_free, dtype=np.int32)

    rows, cols = [], []
    for px in free_pixels:
        y, x = divmod(int(px), RES)
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                if dy == 0 and dx == 0: continue
                ny, nx = y+dy, x+dx
                if 0 <= ny < RES and 0 <= nx < RES:
                    nb_node = pixel_to_node[ny*RES+nx]
                    if nb_node >= 0:
                        rows.append(int(pixel_to_node[px]))
                        cols.append(int(nb_node))

    return free_pixels, pixel_to_node, torch.tensor([rows, cols], dtype=torch.long)

def _make_sub_gt_edge_mask(gt_pixel_arr, free_pixels, sub_ei):
    gt_flat = gt_pixel_arr.flatten()
    src_px  = free_pixels[sub_ei[0].numpy()]
    dst_px  = free_pixels[sub_ei[1].numpy()]
    return torch.tensor(gt_flat[src_px] & gt_flat[dst_px],
                        dtype=torch.float32, device=DEVICE)


# ── cell 6: 데이터 로딩 (node_feats 등 불필요한 부분 생략) ──────────────────
def _find_txt(img_path):
    stem = os.path.basename(img_path).replace('_building_mask.png','')
    direct = os.path.join(TXT_DIR, stem+'_building_places.txt')
    if os.path.exists(direct): return direct
    prefix = stem.replace('-','_').split('_')[0]
    for fn in os.listdir(TXT_DIR):
        if fn.endswith('_building_places.txt') and fn.startswith(prefix):
            return os.path.join(TXT_DIR, fn)
    return None

def load_campus_min(img_path, txt_path):
    img    = Image.open(img_path).convert('L')
    W, H   = img.size
    is_bld = (np.array(img.resize((RES,RES),resample=Image.NEAREST)) > 128).astype(np.float32)

    ridge, _ = _make_common_road_ridge(is_bld)

    ns={}; exec(open(txt_path, encoding='utf-8').read(), ns)
    poly_raw = ns['BUILDING_POLY']; building_ids = list(poly_raw.keys())

    centers = []
    for bid in building_ids:
        pts = poly_raw[bid]
        cy  = np.mean([p[1] for p in pts]) * RES / H
        cx  = np.mean([p[0] for p in pts]) * RES / W
        centers.append((cy, cx))

    bld_nodes = [_nearest_free_node(cy, cx, is_bld) for cy, cx in centers]
    clusters  = _cluster_centers(centers)
    cluster_nodes = []
    for cl in clusters:
        cl_pts = [(centers[i][0], centers[i][1]) for i in cl]
        cy_m   = np.mean([p[0] for p in cl_pts])
        cx_m   = np.mean([p[1] for p in cl_pts])
        max_r  = max(((p[0]-cy_m)**2+(p[1]-cx_m)**2)**0.5 for p in cl_pts) if len(cl_pts)>1 else 0
        if max_r <= CLUSTER_SPLIT_RADIUS:
            final_groups = [cl]
        else:
            sub_cls      = _cluster_centers(cl_pts, eps=CLUSTER_SPLIT_RADIUS)
            final_groups = [[cl[j] for j in sub] for sub in sub_cls]
        for grp in final_groups:
            cy = np.mean([centers[i][0] for i in grp])
            cx = np.mean([centers[i][1] for i in grp])
            node = _nearest_ridge_node(cy, cx, is_bld, ridge)
            if node not in cluster_nodes:
                cluster_nodes.append(node)
    if len(cluster_nodes) <= 1 and len(bld_nodes) > 1:
        cluster_nodes = list(dict.fromkeys(bld_nodes))

    gate_nodes = _select_gate_nodes(cluster_nodes, is_bld, ridge)
    terminals  = list(dict.fromkeys(gate_nodes + cluster_nodes))

    free_pixels, pixel_to_node, sub_ei = build_subgraph(is_bld)

    def px_to_sub(px_list):
        return [int(pixel_to_node[p]) for p in px_list if pixel_to_node[p] >= 0]

    terminal_sub = px_to_sub(terminals)

    slug    = os.path.basename(img_path).replace('_building_mask.png','')
    gt_path = os.path.join(GT_FINAL_DIR, f'{slug}_gt.npz')
    if not os.path.exists(gt_path):
        return None

    gt_arr       = np.load(gt_path)['gt']
    gt_edge_mask = _make_sub_gt_edge_mask(gt_arr, free_pixels, sub_ei)

    ridge_sub  = ridge.flatten()[free_pixels]
    src_n, dst_n = sub_ei[0].numpy(), sub_ei[1].numpy()
    edge_ridge = torch.tensor((ridge_sub[src_n]+ridge_sub[dst_n])*0.5, dtype=torch.float32)

    return {
        'slug': slug, 'sub_ei': sub_ei, 'edge_ridge': edge_ridge,
        'gt_edge_mask': gt_edge_mask, 'terminal_sub': terminal_sub,
        'n_free': len(free_pixels),
    }


# ── 베이스라인 1: Ridge-only ────────────────────────────────────────────────
def ridge_only_pred(c, thr):
    return (c['edge_ridge'] > thr).float()


# ── 베이스라인 2: MST (Steiner tree 근사) ──────────────────────────────────
def mst_pred(c):
    sub_ei = c['sub_ei'].numpy()
    cost   = 1.0 / (c['edge_ridge'].numpy() + 0.1)
    terminals = c['terminal_sub']

    G = nx.Graph()
    for k in range(sub_ei.shape[1]):
        u, v, w = int(sub_ei[0,k]), int(sub_ei[1,k]), float(cost[k])
        if G.has_edge(u, v):
            G[u][v]['weight'] = min(G[u][v]['weight'], w)
        else:
            G.add_edge(u, v, weight=w)

    if len(terminals) < 2:
        return torch.zeros_like(c['edge_ridge'])

    # terminal이 여러 connected component에 걸쳐 있으면 가장 큰 component만 사용
    comps = list(nx.connected_components(G))
    comps.sort(key=len, reverse=True)
    main = comps[0]
    terms_in_main = [t for t in terminals if t in main]
    if len(terms_in_main) < 2:
        return torch.zeros_like(c['edge_ridge'])

    Gsub = G.subgraph(main)
    tree = steiner_tree(Gsub, terms_in_main, weight='weight')
    tree_edges = set()
    for u, v in tree.edges():
        tree_edges.add((u,v)); tree_edges.add((v,u))

    pred = torch.zeros(sub_ei.shape[1])
    for k in range(sub_ei.shape[1]):
        if (int(sub_ei[0,k]), int(sub_ei[1,k])) in tree_edges:
            pred[k] = 1.0
    return pred


def prf(pred, gt):
    pred = pred > 0.5
    gt   = gt > 0.5
    tp = (pred & gt).sum().item()
    p_sum = pred.sum().item()
    g_sum = gt.sum().item()
    precision = tp / p_sum if p_sum > 0 else 0.0
    recall    = tp / g_sum if g_sum > 0 else 0.0
    f1 = 2*precision*recall/(precision+recall) if (precision+recall) > 0 else 0.0
    return precision, recall, f1


def main():
    campuses = []
    for img_path in sorted(glob.glob(os.path.join(IMG_DIR, '*_building_mask.png'))):
        txt = _find_txt(img_path)
        if not txt: continue
        slug = os.path.basename(img_path).replace('_building_mask.png','')
        if not os.path.exists(os.path.join(GT_FINAL_DIR, f'{slug}_gt.npz')):
            continue
        try:
            c = load_campus_min(img_path, txt)
        except Exception as e:
            print(f'skip {slug}: {e}')
            continue
        if c is not None:
            campuses.append(c)

    print(f'GT 있는 캠퍼스 {len(campuses)}개 로드 완료')

    # ── Ridge-only: 임계값 스윕 ──────────────────────────────────────────
    print('\n=== Ridge-only ===')
    for thr in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]:
        ps, rs, fs = [], [], []
        for c in campuses:
            pred = ridge_only_pred(c, thr)
            p, r, f = prf(pred, c['gt_edge_mask'])
            ps.append(p); rs.append(r); fs.append(f)
        print(f'thr={thr:.1f}  precision={np.mean(ps):.4f}  recall={np.mean(rs):.4f}  f1={np.mean(fs):.4f}')

    # ── MST ──────────────────────────────────────────────────────────────
    print('\n=== MST (Steiner tree) ===')
    ps, rs, fs = [], [], []
    for c in campuses:
        pred = mst_pred(c)
        p, r, f = prf(pred, c['gt_edge_mask'])
        ps.append(p); rs.append(r); fs.append(f)
        print(f"  {c['slug']:45s} precision={p:.4f}  recall={r:.4f}  f1={f:.4f}")
    print(f'\nMST 평균  precision={np.mean(ps):.4f}  recall={np.mean(rs):.4f}  f1={np.mean(fs):.4f}')


if __name__ == '__main__':
    main()
