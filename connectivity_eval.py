"""6개 모델(Trivial/MST/Ridge-only/SAGE/MLP/V28)의 Connectivity Rate, Edge Density 계산.

val+test 44개 캠퍼스(GT 유무와 무관, 예측 결과 + terminal 위치만 필요)에 대해:
- Edge Density    = 예측 도로 엣지 수 / 전체 free 엣지 수
- Connectivity Rate = 모든 terminal(게이트+건물군) 중,
                       인접 엣지 하나라도 예측된 노드의 비율

GAT/MLP/SAGE는 저장된 체크포인트로 추론(학습 불필요).
"""
import os, glob, random
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
from PIL import Image
from scipy.ndimage import distance_transform_edt, maximum_filter, gaussian_filter
import networkx as nx
from networkx.algorithms.approximation import steiner_tree

RES = 100
N = RES * RES
DEVICE = torch.device('cpu')
NODE_DIM = 9
GAT_HEADS = 4
ATTN_DROPOUT = 0.05
EDGE_THR = 0.50
RIDGE_THR = 0.10  # 9-2 표에서 채택한 Ridge-only 최적 임계값

CLUSTER_EPS          = 13.0
CLUSTER_SPLIT_RADIUS = 15.0
ROAD_CLEAR_MIN    = 3.0
ROAD_CLEAR_MAX    = 24.0
RIDGE_FILTER_SIZE = 7
N_GATES       = 2
GATE_MIN_DIST = 14

IMG_DIR      = 'collegemap/images'
TXT_DIR      = 'collegemap/txt'
GT_FINAL_DIR = 'collegemap/gt_masks_final'


# ── 헬퍼 (baseline_eval.py와 동일) ──────────────────────────────────────────
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

def _dist_to_px(query_pixels, target_pixels, tau):
    if len(target_pixels) == 0:
        return np.zeros(len(query_pixels), dtype=np.float32)
    qy = query_pixels // RES; qx = query_pixels % RES
    feats = []
    for t in target_pixels:
        ty, tx = divmod(int(t), RES)
        feats.append(np.sqrt((qy-ty)**2 + (qx-tx)**2))
    dmin = np.stack(feats, axis=0).min(axis=0)
    return np.exp(-dmin / tau).astype(np.float32)

def _direction_to_targets(free_pixels, target_nodes):
    n_free = len(free_pixels)
    if len(target_nodes) == 0:
        return np.zeros(n_free, np.float32), np.zeros(n_free, np.float32)
    py = (free_pixels // RES).astype(np.float32)
    px = (free_pixels  % RES).astype(np.float32)
    min_dy = np.zeros(n_free, np.float32); min_dx = np.zeros(n_free, np.float32)
    min_d2 = np.full(n_free, np.inf, np.float32)
    for t in target_nodes:
        ty, tx = divmod(int(t), RES)
        dy = ty - py; dx = tx - px
        d2 = dy*dy + dx*dx
        closer = d2 < min_d2
        min_d2[closer] = d2[closer]; min_dy[closer] = dy[closer]; min_dx[closer] = dx[closer]
    dist = np.sqrt(min_d2) + 1e-6
    return (min_dy / dist).astype(np.float32), (min_dx / dist).astype(np.float32)

def _find_txt(img_path):
    stem = os.path.basename(img_path).replace('_building_mask.png','')
    direct = os.path.join(TXT_DIR, stem+'_building_places.txt')
    if os.path.exists(direct): return direct
    prefix = stem.replace('-','_').split('_')[0]
    for fn in os.listdir(TXT_DIR):
        if fn.endswith('_building_places.txt') and fn.startswith(prefix):
            return os.path.join(TXT_DIR, fn)
    return None


# ── load_campus (cell 6, flip=None, node_feats 포함) ───────────────────────
def load_campus(img_path, txt_path):
    img    = Image.open(img_path).convert('L')
    W, H   = img.size
    is_bld = (np.array(img.resize((RES,RES),resample=Image.NEAREST)) > 128).astype(np.float32)

    ridge, dist = _make_common_road_ridge(is_bld)
    dist_n = (dist / (dist.max()+1e-6)).astype(np.float32)

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
    n_free = len(free_pixels)

    def px_to_sub(px_list):
        return [int(pixel_to_node[p]) for p in px_list if pixel_to_node[p] >= 0]

    terminal_sub = px_to_sub(terminals)

    xs = (free_pixels % RES).astype(np.float32) / RES
    ys = (free_pixels // RES).astype(np.float32) / RES
    ridge_sub  = ridge.flatten()[free_pixels]
    dist_n_sub = dist_n.flatten()[free_pixels]

    cluster_sub = px_to_sub(cluster_nodes)
    cluster_indicator = np.zeros(n_free, dtype=np.float32)
    for nd in cluster_sub:
        if 0 <= nd < n_free: cluster_indicator[nd] = 1.0

    d_cluster = _dist_to_px(free_pixels, cluster_nodes, tau=12.0)
    d_gate    = _dist_to_px(free_pixels, gate_nodes,    tau=15.0)
    dy_cluster, dx_cluster = _direction_to_targets(free_pixels, cluster_nodes)

    node_feats = np.stack([ridge_sub, dist_n_sub, xs, ys,
                           cluster_indicator, d_cluster, d_gate,
                           dy_cluster, dx_cluster], axis=1)

    src_n, dst_n = sub_ei[0].numpy(), sub_ei[1].numpy()
    edge_ridge = torch.tensor((ridge_sub[src_n]+ridge_sub[dst_n])*0.5, dtype=torch.float32)

    slug = os.path.basename(img_path).replace('_building_mask.png','')
    has_gt = os.path.exists(os.path.join(GT_FINAL_DIR, f'{slug}_gt.npz'))

    return {
        'slug': slug, 'sub_ei': sub_ei, 'edge_ridge': edge_ridge,
        'terminal_sub': terminal_sub, 'n_free': n_free,
        'node_feats': torch.tensor(node_feats, dtype=torch.float32),
        'gt_source': 'osm' if has_gt else 'test_no_osm',
    }


# ── 모델 정의 ────────────────────────────────────────────────────────────
class MHGATLayer(nn.Module):
    def __init__(self, in_dim, out_dim, heads=4, dropout=0.0):
        super().__init__()
        assert out_dim % heads == 0
        self.heads=heads; self.dh=out_dim//heads
        self.W    = nn.Linear(in_dim, out_dim, bias=False)
        self.a_s  = nn.Parameter(torch.empty(heads,self.dh))
        self.a_d  = nn.Parameter(torch.empty(heads,self.dh))
        self.norm = nn.LayerNorm(out_dim)
        self.skip = nn.Linear(in_dim,out_dim,bias=False) if in_dim!=out_dim else nn.Identity()
        self.drop = nn.Dropout(dropout)
        nn.init.xavier_uniform_(self.W.weight,gain=1.414)
        nn.init.xavier_normal_(self.a_s.unsqueeze(0))
        nn.init.xavier_normal_(self.a_d.unsqueeze(0))

    def forward(self, x, ei):
        n=x.size(0); src,dst=ei[0],ei[1]
        h=self.W(x).view(n,self.heads,self.dh)
        e=F.leaky_relu((h[src]*self.a_s).sum(-1)+(h[dst]*self.a_d).sum(-1),0.2)
        e_exp=torch.exp(e-e.max())
        denom=torch.zeros(n,self.heads,device=x.device)
        denom.scatter_add_(0,src.unsqueeze(1).expand(-1,self.heads),e_exp)
        alpha=self.drop(e_exp/(denom[src]+1e-8))
        msg=alpha.unsqueeze(-1)*h[dst]
        agg=torch.zeros(n,self.heads,self.dh,device=x.device)
        agg.scatter_add_(0,src[:,None,None].expand_as(msg),msg)
        return self.norm(F.elu(agg.view(n,-1))+self.skip(x))

def _edge_feat(h3, feats, ei):
    src, dst = ei[0], ei[1]
    hs, hd = h3[src], h3[dst]
    raw_edge = torch.stack([
        (feats[src,0]+feats[dst,0])*0.5,
        (src!=dst).float(),
    ],dim=1)
    return torch.cat([hs,hd,torch.abs(hs-hd),hs*hd,raw_edge],dim=1)

class SubgraphGAT(nn.Module):
    def __init__(self):
        super().__init__()
        self.gat1 = MHGATLayer(NODE_DIM, 64, GAT_HEADS, ATTN_DROPOUT)
        self.gat2 = MHGATLayer(64,       64, GAT_HEADS, ATTN_DROPOUT)
        self.gat3 = MHGATLayer(64,       64, GAT_HEADS, ATTN_DROPOUT)
        self.edge_head = nn.Sequential(
            nn.Linear(64*4+2, 128), nn.ReLU(), nn.Dropout(0.08),
            nn.Linear(128, 32), nn.ReLU(), nn.Linear(32, 1))
    def forward(self, feats, ei):
        h1=self.gat1(feats,ei); h2=self.gat2(h1,ei); h3=self.gat3(h2,ei)
        edge_logits=self.edge_head(_edge_feat(h3,feats,ei)).squeeze(-1).float()
        edge_logits=torch.nan_to_num(edge_logits,nan=0.,posinf=20.,neginf=-20.).clamp(-20.,20.)
        return torch.sigmoid(edge_logits)

class SubgraphMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.node_mlp = nn.Sequential(
            nn.Linear(NODE_DIM, 64), nn.LayerNorm(64), nn.ELU(), nn.Dropout(ATTN_DROPOUT),
            nn.Linear(64, 64), nn.LayerNorm(64), nn.ELU(), nn.Dropout(ATTN_DROPOUT),
            nn.Linear(64, 64), nn.LayerNorm(64), nn.ELU())
        self.edge_head = nn.Sequential(
            nn.Linear(64*4+2, 128), nn.ReLU(), nn.Dropout(0.08),
            nn.Linear(128, 32), nn.ReLU(), nn.Linear(32, 1))
    def forward(self, feats, ei):
        h3 = self.node_mlp(feats)
        edge_logits=self.edge_head(_edge_feat(h3,feats,ei)).squeeze(-1).float()
        edge_logits=torch.nan_to_num(edge_logits,nan=0.,posinf=20.,neginf=-20.).clamp(-20.,20.)
        return torch.sigmoid(edge_logits)

class SAGEConv(nn.Module):
    def __init__(self, in_dim, out_dim, dropout=0.0):
        super().__init__()
        self.W = nn.Linear(in_dim * 2, out_dim)
        self.norm = nn.LayerNorm(out_dim)
        self.drop = nn.Dropout(dropout)
    def forward(self, x, ei):
        n = x.size(0); src, dst = ei[0], ei[1]
        msg = x[dst]
        agg = torch.zeros(n, x.size(1), device=x.device)
        agg.scatter_add_(0, src.unsqueeze(1).expand(-1, x.size(1)), msg)
        deg = torch.zeros(n, 1, device=x.device)
        deg.scatter_add_(0, src.unsqueeze(1), torch.ones(src.size(0), 1, device=x.device))
        agg_mean = agg / (deg + 1e-8)
        out = torch.cat([x, agg_mean], dim=1)
        out = F.elu(self.W(out))
        return self.norm(self.drop(out))

class SubgraphSAGE(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = SAGEConv(NODE_DIM, 64, ATTN_DROPOUT)
        self.conv2 = SAGEConv(64,       64, ATTN_DROPOUT)
        self.conv3 = SAGEConv(64,       64, ATTN_DROPOUT)
        self.edge_head = nn.Sequential(
            nn.Linear(64*4+2, 128), nn.ReLU(), nn.Dropout(0.08),
            nn.Linear(128, 32), nn.ReLU(), nn.Linear(32, 1))
    def forward(self, feats, ei):
        h1 = self.conv1(feats, ei); h2 = self.conv2(h1, ei); h3 = self.conv3(h2, ei)
        edge_logits=self.edge_head(_edge_feat(h3,feats,ei)).squeeze(-1).float()
        edge_logits=torch.nan_to_num(edge_logits,nan=0.,posinf=20.,neginf=-20.).clamp(-20.,20.)
        return torch.sigmoid(edge_logits)


def load_ckpt(model, path):
    ck = torch.load(path, map_location=DEVICE)
    model.load_state_dict(ck['model'] if isinstance(ck, dict) and 'model' in ck else ck)
    model.eval()
    return model


# ── MST (Steiner tree 근사, baseline_eval.py와 동일) ───────────────────────
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
        return torch.zeros(sub_ei.shape[1])

    comps = list(nx.connected_components(G))
    comps.sort(key=len, reverse=True)
    main = comps[0]
    terms_in_main = [t for t in terminals if t in main]
    if len(terms_in_main) < 2:
        return torch.zeros(sub_ei.shape[1])

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


# ── 지표 계산 ────────────────────────────────────────────────────────────
def edge_density(pred, total):
    return pred.sum().item() / total

def connectivity_rate(pred, sub_ei, terminal_sub):
    if not terminal_sub:
        return 1.0
    src, dst = sub_ei[0], sub_ei[1]
    connected = 0
    for t in terminal_sub:
        incident = (src == t) | (dst == t)
        if incident.any() and pred[incident].sum() > 0:
            connected += 1
    return connected / len(terminal_sub)


def main():
    campuses = []
    for img_path in sorted(glob.glob(os.path.join(IMG_DIR, '*_building_mask.png'))):
        txt = _find_txt(img_path)
        if not txt: continue
        try:
            c = load_campus(img_path, txt)
        except Exception as e:
            print(f'skip {os.path.basename(img_path)}: {e}')
            continue
        campuses.append(c)

    osm_orig = [c for c in campuses if c['gt_source']=='osm']
    test_campuses = [c for c in campuses if c['gt_source']=='test_no_osm']

    random.seed(42); random.shuffle(osm_orig)
    n_val = max(1, len(osm_orig)//7)
    val_campuses = osm_orig[:n_val]

    eval_campuses = val_campuses + test_campuses
    print(f'평가 캠퍼스: val {len(val_campuses)}개 + test {len(test_campuses)}개 = {len(eval_campuses)}개\n')

    gat  = SubgraphGAT();  load_ckpt(gat,  'v28a/v28_subgraph_gnn.pth')
    mlp  = SubgraphMLP();  load_ckpt(mlp,  'v28_mlp/v28a_subgraph_mlp.pth')
    sage = SubgraphSAGE(); load_ckpt(sage, 'v28a_sage/v28a_subgraph_sage.pth')

    results = {name: {'density':[], 'conn':[]} for name in
               ['Trivial','MST','Ridge-only','SAGE','MLP','V28(GAT)']}

    with torch.no_grad():
        for c in eval_campuses:
            sub_ei = c['sub_ei']; total = sub_ei.shape[1]; term = c['terminal_sub']

            pred_trivial = torch.ones(total)
            pred_ridge   = (c['edge_ridge'] > RIDGE_THR).float()
            pred_mst     = mst_pred(c)
            pred_gat     = (gat (c['node_feats'], sub_ei) > EDGE_THR).float()
            pred_mlp     = (mlp (c['node_feats'], sub_ei) > EDGE_THR).float()
            pred_sage    = (sage(c['node_feats'], sub_ei) > EDGE_THR).float()

            for name, pred in [('Trivial',pred_trivial), ('MST',pred_mst),
                                ('Ridge-only',pred_ridge), ('SAGE',pred_sage),
                                ('MLP',pred_mlp), ('V28(GAT)',pred_gat)]:
                results[name]['density'].append(edge_density(pred, total))
                results[name]['conn'].append(connectivity_rate(pred, sub_ei, term))

    print(f'{"모델":15s} {"Connectivity Rate":>20s} {"Edge Density":>15s}')
    for name, r in results.items():
        print(f'{name:15s} {np.mean(r["conn"])*100:19.1f}% {np.mean(r["density"])*100:14.1f}%')


if __name__ == '__main__':
    main()
