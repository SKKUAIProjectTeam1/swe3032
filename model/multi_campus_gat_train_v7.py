"""
multi_campus_gat_train_v7.py  —  Multi-Campus GAT Road Designer (V23 + 통행량 위계 GT)

V2~V6의 이진 GT 마스크(corridor/OSM/지름길/spur)는 모두 "여기 도로가 있어야
함"만 말할 뿐 도로의 위계(줄기/가지/잔가지)를 말하지 않음 → 어떤 GT를 추가해도
결과가 "지렁이"형(균일 굵기로 구불구불 한 줄)에서 벗어나지 못했음.

실제 캠퍼스 도로의 위계는 통행량에서 나옴: 모든 이동이 공유하는 구간은 줄기,
한두 건물만 쓰는 끝 구간은 잔가지. 이를 직접 모사:

  ✓ _traffic_edge_target — 건물 쌍 최단경로를 전부(샘플링) 누적해 각 칸의
    "통행량"을 세고, log 스케일로 0~1 연속 타깃화. 줄기≈1, 잔가지≈0.1.
    이진 마스크 대신 연속 타깃에 soft-target BCE → ew가 통행량 위계를 학습.
  ✓ OSM 실제 도로 마스크가 있는 캠퍼스는 해당 엣지 타깃을 0.5 이상으로 보정
    (실제 도로 기하 유지).
  ✓ ridge_loss는 통행량 GT가 있는 엣지에서 면제 — GT 경로가 skeleton을
    벗어나 곧게 가도 벌점 없음 ("구비구비" 완화).
  ✗ _auto_gt_edge_mask / _shortcut_gt_edge_mask / _spur_gt_edge_mask 제거
    (통행량 GT가 셋 모두를 포괄: 공유 corridor=고통행, 지름길/spur=저통행 경로)

train/test split 유지: TEST_CAMPUSES 5개는 학습에서 제외(hold-out).

실행:
  python model/multi_campus_gat_train_v7.py                      # 학습 → hold-out 시각화
  python model/multi_campus_gat_train_v7.py --epochs 20          # epoch 수 조절 (시험 실행)
  python model/multi_campus_gat_train_v7.py --load               # 저장 모델 로드 → 시각화
  python model/multi_campus_gat_train_v7.py --load --infer sungkyunkwan_university
"""

import argparse, glob, os, signal, sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from PIL import Image
from scipy.ndimage import distance_transform_edt, binary_dilation
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra, minimum_spanning_tree
from skimage.morphology import skeletonize

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

# ── 설정 ─────────────────────────────────────────────────────────────────────
RES           = 100          # 격자 해상도 (RES × RES 노드)
N             = RES * RES
NODE_DIM      = 5            # [is_building, ridge, dist_norm, x_norm, y_norm]
GAT_HEADS     = 4
DIFF_STEPS    = 25           # 확산 스텝 수 (건물 간 최대 거리 커버)
EPOCHS        = 300
LR            = 3e-4
WARMUP_EPOCHS = 50
ATTN_DROPOUT  = 0.1
N_GATES       = 2            # 캠퍼스당 추천 관문 수
GATE_MIN_DIST = 14           # 관문 간 최소 L1 거리 (격자 단위)
DEVICE        = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

ROOT     = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
IMG_DIR  = os.path.join(ROOT, 'collegemap', 'images')
TXT_DIR  = os.path.join(ROOT, 'collegemap', 'txt')
ROAD_DIR = os.path.join(ROOT, 'collegemap', 'road_masks')
OUT_DIR  = os.path.join(ROOT, 'output')
CKPT     = os.path.join(OUT_DIR, 'multi_campus_gat_v7.pth')
os.makedirs(OUT_DIR, exist_ok=True)

GT_LOSS_WEIGHT = 20.   # GT 도로(OSM 마스크 또는 자동 corridor) 구간 BCE 가중치
SPARSE_WEIGHT  = 4.    # 총 도로량 억제 강도 (V4: 8 → 완화, 분기 허용)
DEGREE_WEIGHT  = 0.5   # 교차점(분기) 억제 강도 (V4: 1.5 → 완화)
DEGREE_THRESH  = 8.    # 패널티 없이 허용하는 교차점 차수 (V4: 5 → 완화)

# 학습에서 제외하고 일반화 평가용으로 hold-out하는 캠퍼스 (V21 size_test와 동일 5개)
TEST_CAMPUSES = {
    'kwangwoon_university',
    'sahmyook_university',
    'hanyang_university_erica',
    'kangwon_national_university',
    'national_university_of_singapore',
}

# ── 고정 그래프 토폴로지 (모든 캠퍼스 공통) ───────────────────────────────────
def _build_edges(res: int) -> torch.Tensor:
    """RES×RES 격자: 8방향 엣지 + self-loop"""
    edges = []
    for y in range(res):
        for x in range(res):
            s = y * res + x
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    if dy == 0 and dx == 0:
                        continue
                    nx_, ny_ = x + dx, y + dy
                    if 0 <= nx_ < res and 0 <= ny_ < res:
                        edges.append((s, ny_ * res + nx_))
    edges += [(i, i) for i in range(res * res)]
    return torch.tensor(edges, dtype=torch.long).t().contiguous()

EDGE_INDEX = _build_edges(RES).to(DEVICE)

# 관문 후보: 외곽 경계 노드 (코너 10칸 제외)
_bdy = []
for i in range(10, RES - 10):
    _bdy += [i, (RES - 1) * RES + i, i * RES, i * RES + (RES - 1)]
BOUNDARY   = sorted(set(_bdy))
BOUNDARY_T = torch.tensor(BOUNDARY, dtype=torch.long).to(DEVICE)


# ── 데이터 로딩 ───────────────────────────────────────────────────────────────
def _find_txt(img_path: str) -> str | None:
    """이미지 파일명에 대응하는 BUILDING_POLY txt 파일 경로 반환"""
    stem   = os.path.basename(img_path).replace('_building_mask.png', '')
    direct = os.path.join(TXT_DIR, stem + '_building_places.txt')
    if os.path.exists(direct):
        return direct
    # fuzzy: 첫 단어 prefix 매칭 (예: chung-ang-univ → chung-ang_university...)
    prefix = stem.replace('-', '_').split('_')[0]
    for fn in os.listdir(TXT_DIR):
        if fn.endswith('_building_places.txt') and fn.startswith(prefix):
            return os.path.join(TXT_DIR, fn)
    return None


def _traffic_edge_target(is_bld: np.ndarray, building_nodes: list[int],
                         max_pairs: int = 4000) -> torch.Tensor | None:
    """건물 쌍 최단경로 누적 통행량 → 엣지별 연속 GT 타깃 (0~1).

    실제 캠퍼스 도로의 줄기/가지/잔가지 위계는 통행량에서 나온다:
    모든 이동이 공유하는 구간은 줄기, 한두 건물만 쓰는 끝은 잔가지.
    건물 쌍(최대 max_pairs개 샘플링) 사이 자유공간 최단경로를 전부 누적해
    각 칸의 통과 횟수를 세고, log 스케일로 0~1 정규화한다.
    줄기 칸 ≈ 1, 잔가지 칸 ≈ 0.1, 경로가 안 지나는 칸 = 0.
    """
    if len(building_nodes) < 2:
        return None

    src_all, dst_all = EDGE_INDEX[0].cpu().numpy(), EDGE_INDEX[1].cpu().numpy()
    free  = (is_bld.flatten() == 0)
    valid = (src_all != dst_all) & free[src_all] & free[dst_all]
    s, d  = src_all[valid], dst_all[valid]
    graph = csr_matrix((np.ones(len(s)), (s, d)), shape=(N, N))

    B = list(building_nodes)
    dmat, pred = dijkstra(graph, indices=B, return_predecessors=True, unweighted=True)

    pairs = [(i, j) for i in range(len(B)) for j in range(i + 1, len(B))]
    if len(pairs) > max_pairs:
        rng   = np.random.default_rng(0)
        sel   = rng.choice(len(pairs), max_pairs, replace=False)
        pairs = [pairs[k] for k in sel]

    count = np.zeros(N, dtype=np.float64)
    for i, j in pairs:
        node = B[j]
        if not np.isfinite(dmat[i, node]):
            continue
        while node != B[i] and node >= 0:
            count[node] += 1.
            node = pred[i, node]
        count[B[i]] += 1.
    count[~free] = 0.
    if count.max() <= 0:
        return None

    t = np.log1p(count) / np.log1p(count.max())   # 통행량 위계: 줄기≈1, 잔가지≈0.1
    t_edge = np.minimum(t[src_all], t[dst_all])    # 양 끝 모두 경로 위인 엣지만 > 0
    return torch.tensor(t_edge, dtype=torch.float32).to(DEVICE)


def _osm_gt_edge_mask(is_bld: np.ndarray, name: str) -> torch.Tensor | None:
    """OSM에서 수집한 실제 도로 마스크(road_masks/*.npy)를 GT 엣지 마스크로 변환.

    마스크는 100×100 soft 래스터(anti-aliased) — >0 이진화로 도로 칸을 얻고,
    건물 칸과 겹치는 부분은 제외(hard masking과 일관성 유지). 도로 칸 양 끝에
    걸친 엣지를 GT로 표시한다. 마스크 파일이 없으면 None (호출측에서 자동
    corridor로 폴백).
    """
    path = os.path.join(ROAD_DIR, f'{name}_road_mask.npy')
    if not os.path.exists(path):
        return None
    road = np.load(path)
    if road.shape != (RES, RES):
        return None
    road_b = (road > 0) & (is_bld == 0)
    flat = road_b.flatten()
    src_all, dst_all = EDGE_INDEX[0].cpu().numpy(), EDGE_INDEX[1].cpu().numpy()
    mask = flat[src_all] & flat[dst_all]
    if not mask.any():
        return None
    return torch.tensor(mask, dtype=torch.bool).to(DEVICE)


def _nearest_road_node(cy: float, cx: float, is_bld_grid: np.ndarray) -> int:
    """건물 중심(cy, cx)에서 가장 가까운 비건물 노드 인덱스 반환"""
    non_bld_yx = np.argwhere(is_bld_grid == 0)   # (K, 2) [y, x]
    if len(non_bld_yx) == 0:
        return int(cy) * RES + int(cx)
    dists = (non_bld_yx[:, 0] - cy) ** 2 + (non_bld_yx[:, 1] - cx) ** 2
    best  = non_bld_yx[np.argmin(dists)]
    return int(best[0]) * RES + int(best[1])


def load_campus(img_path: str, txt_path: str) -> dict:
    name = os.path.basename(img_path).replace('_building_mask.png', '')

    # 건물 마스크 이미지 → RES×RES
    img    = Image.open(img_path).convert('L')
    W, H   = img.size
    small  = img.resize((RES, RES), resample=Image.NEAREST)
    is_bld = (np.array(small) > 128).astype(np.float32)   # (RES, RES)

    dist   = distance_transform_edt(1 - is_bld)
    d_max  = dist.max() if dist.max() > 0 else 1.0
    dist_n = (dist / d_max).astype(np.float32)

    # ridge: 자유공간(건물에서 1.5칸 이상)의 중심선(skeleton)을 1칸 팽창.
    # "건물 주변 띠 전체"를 보상하면 도로가 건물 외곽을 감싸는 테두리가 되므로,
    # 건물 사이 한가운데 능선만 보상해 도로가 복도 중앙을 따라가게 유도
    skel  = skeletonize(dist > 1.5)
    ridge = binary_dilation(skel, iterations=1).astype(np.float32)

    # 노드 피처 5채널: [is_building, ridge, dist_norm, x_norm, y_norm]
    xs = np.tile(np.arange(RES),          RES).astype(np.float32) / RES
    ys = np.repeat(np.arange(RES), RES       ).astype(np.float32) / RES
    node_feats = np.stack(
        [is_bld.flatten(), ridge.flatten(), dist_n.flatten(), xs, ys], axis=1
    ).astype(np.float32)                                   # (N, 5)

    # hard masking용: 건물 노드에 닿는 엣지 마스크
    ib_flat  = torch.tensor(is_bld.flatten(), dtype=torch.float32).to(DEVICE)
    src, dst = EDGE_INDEX[0], EDGE_INDEX[1]
    bld_mask = (ib_flat[src] > 0) | (ib_flat[dst] > 0)

    # BUILDING_POLY 로드 → 건물별 가장 가까운 비건물 노드
    # (hard masking 하에서 건물 내부 노드는 도달 불가이므로)
    ns = {}
    with open(txt_path, encoding='utf-8') as f:
        exec(f.read(), ns)
    poly = ns['BUILDING_POLY']

    bld_nodes = list({
        _nearest_road_node(
            np.mean([p[1] for p in pts]) * RES / H,
            np.mean([p[0] for p in pts]) * RES / W,
            is_bld
        )
        for pts in poly.values()
    })

    # GT: 통행량 위계 타깃 (줄기≈1 ~ 잔가지≈0.1, 연속값)
    gt_target = _traffic_edge_target(is_bld, bld_nodes)
    gt_src    = 'traffic'

    # OSM 실제 도로 마스크가 있으면 해당 엣지 타깃을 0.5 이상으로 보정
    # (실제 도로 기하를 유지하면서 통행량 위계와 결합)
    osm_mask = _osm_gt_edge_mask(is_bld, name)
    if osm_mask is not None:
        boost     = osm_mask.float() * 0.5
        gt_target = boost if gt_target is None else torch.maximum(gt_target, boost)
        gt_src   += '+osm'

    return {
        'node_feats':     torch.tensor(node_feats,       dtype=torch.float32).to(DEVICE),
        'is_building':    ib_flat,
        'bld_mask':       bld_mask,
        'ridge':          torch.tensor(ridge.flatten(),  dtype=torch.float32).to(DEVICE),
        'building_nodes': torch.tensor(bld_nodes,        dtype=torch.long   ).to(DEVICE),
        'gt_edge_target': gt_target,
        'gt_src':         gt_src,
        'poly': poly, 'name': name, 'W': W, 'H': H,
    }


def load_all() -> list[dict]:
    imgs = sorted(glob.glob(os.path.join(IMG_DIR, '*_building_mask.png')))
    if not imgs:
        sys.exit(f'[ERROR] 이미지 없음: {IMG_DIR}')
    print(f'▶ {len(imgs)}개 캠퍼스 이미지 로드 중...')
    campuses = []
    for img_path in imgs:
        txt = _find_txt(img_path)
        if txt is None:
            print(f'  [SKIP] txt 없음: {os.path.basename(img_path)}')
            continue
        try:
            c = load_campus(img_path, txt)
            campuses.append(c)
            print(f'  ✓ {c["name"]:60s} ({len(c["building_nodes"])} 건물, GT={c["gt_src"]})')
        except Exception as e:
            print(f'  ✗ {os.path.basename(img_path)}: {e}')
    n_osm = sum(1 for c in campuses if 'osm' in c['gt_src'])
    print(f'  → 총 {len(campuses)}개 로드 완료 (통행량 GT 전체 / OSM 보정 {n_osm}개)\n')
    return campuses


# ── Multi-Head Sparse GAT 레이어 ──────────────────────────────────────────────
class MHGATLayer(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, heads: int = 4, dropout: float = 0.0):
        super().__init__()
        assert out_dim % heads == 0
        self.heads = heads
        self.dh    = out_dim // heads
        self.W     = nn.Linear(in_dim, out_dim, bias=False)
        self.a_s   = nn.Parameter(torch.empty(heads, self.dh))
        self.a_d   = nn.Parameter(torch.empty(heads, self.dh))
        self.norm  = nn.LayerNorm(out_dim)
        self.skip  = nn.Linear(in_dim, out_dim, bias=False) if in_dim != out_dim else nn.Identity()
        self.drop  = nn.Dropout(dropout)
        nn.init.xavier_uniform_(self.W.weight, gain=1.414)
        nn.init.xavier_normal_(self.a_s.unsqueeze(0))
        nn.init.xavier_normal_(self.a_d.unsqueeze(0))

    def forward(self, x: torch.Tensor, ei: torch.Tensor) -> torch.Tensor:
        n = x.size(0)
        src, dst = ei[0], ei[1]
        h = self.W(x).view(n, self.heads, self.dh)
        e = F.leaky_relu(
            (h[src] * self.a_s).sum(-1) +
            (h[dst] * self.a_d).sum(-1), 0.2
        )
        e_exp = torch.exp(e - e.max())
        denom = torch.zeros(n, self.heads, device=x.device)
        denom.scatter_add_(0, src.unsqueeze(1).expand(-1, self.heads), e_exp)
        alpha = self.drop(e_exp / (denom[src] + 1e-8))
        msg   = alpha.unsqueeze(-1) * h[dst]
        agg   = torch.zeros(n, self.heads, self.dh, device=x.device)
        agg.scatter_add_(0, src[:, None, None].expand_as(msg), msg)
        return self.norm(F.elu(agg.view(n, -1)) + self.skip(x))


# ── MultiCampusRoadGAT ────────────────────────────────────────────────────────
class MultiCampusRoadGAT(nn.Module):
    """
    Campus-agnostic: 어떤 캠퍼스든 동일 가중치로 처리.
    입력: node_feats (N, 5), bld_mask (E,)
    출력: edge_weights (E,), gate_scores (|BOUNDARY|,)
    """
    def __init__(self):
        super().__init__()
        self.gat1 = MHGATLayer(NODE_DIM, 64, heads=GAT_HEADS, dropout=ATTN_DROPOUT)
        self.gat2 = MHGATLayer(64,       64, heads=GAT_HEADS, dropout=ATTN_DROPOUT)
        self.gat3 = MHGATLayer(64,       32, heads=GAT_HEADS, dropout=ATTN_DROPOUT)

        # edge_head: cat(h2[src], h2[dst]) = (E, 128) → 도로 확률
        self.edge_head = nn.Sequential(
            nn.Linear(128, 32), nn.ReLU(), nn.Dropout(0.1), nn.Linear(32, 1)
        )
        # gate_head: h3[boundary] = (|B|, 32) → 관문 점수
        self.gate_head = nn.Linear(32, 1)

        for m in self.modules():
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.zeros_(m.bias)

    def forward(self, feats: torch.Tensor, ei: torch.Tensor, bld_mask: torch.Tensor):
        h1 = self.gat1(feats, ei)                                   # (N, 64)
        h2 = self.gat2(h1,    ei)                                   # (N, 64)
        h3 = self.gat3(h2,    ei)                                   # (N, 32)
        src, dst = ei[0], ei[1]
        raw = self.edge_head(torch.cat([h2[src], h2[dst]], dim=1)).squeeze(-1)
        ew  = torch.sigmoid(raw) * (~bld_mask).float()   # hard masking: 건물 관통 차단
        gs  = torch.sigmoid(self.gate_head(h3[BOUNDARY_T]).squeeze(-1))
        return ew, gs


# ── 관문 선택 (학습 loss와 시각화에서 공유) ───────────────────────────────────
def _pick_gates(gs: torch.Tensor) -> list[int]:
    """gate 점수 상위에서 최소 거리 제약을 지키며 N_GATES개 노드 greedy 선택.
    반환: BOUNDARY 리스트에 대한 인덱스"""
    order  = torch.argsort(gs, descending=True).cpu().numpy()
    chosen = []
    for gi in order:
        gx = BOUNDARY[gi] % RES
        gy = BOUNDARY[gi] // RES
        if all(abs(gx - BOUNDARY[p] % RES) + abs(gy - BOUNDARY[p] // RES) >= GATE_MIN_DIST
               for p in chosen):
            chosen.append(int(gi))
        if len(chosen) == N_GATES:
            break
    return chosen


# ── 그래프 확산 연결성 손실 ───────────────────────────────────────────────────
def diffusion_conn_loss(
    ew:             torch.Tensor,   # (E,)  엣지 가중치
    source_nodes:   torch.Tensor,   # (B,)  건물 인접 노드 + 관문 노드 인덱스
    ei:             torch.Tensor,   # (2,E)
    T:              int = DIFF_STEPS,
    weight:         float = 30.0,
) -> torch.Tensor:
    """
    각 소스 노드에서 신호를 발사해 T번 전파.
    다른 소스 노드에 신호가 도달하지 않으면 패널티 → 실제 경로 존재를 강제.

    신호 업데이트:
      msg[dst] += norm_ew * signals[src]
      signals  = clamp(signals + msg, 0, 1)

    norm_ew: source degree로 row-normalize → 신호가 발산하지 않도록
    """
    src, dst = ei[0], ei[1]
    n_src    = source_nodes.size(0)
    if n_src <= 1:
        return ew.sum() * 0.

    deg     = torch.zeros(N, device=DEVICE).scatter_add_(0, src, ew)
    norm_ew = ew / (deg[src] + 1e-8)                        # (E,)

    signals = torch.zeros(N, n_src, device=DEVICE)
    signals[source_nodes, torch.arange(n_src, device=DEVICE)] = 1.0

    for _ in range(T):
        msg = torch.zeros(N, n_src, device=DEVICE)
        msg.scatter_add_(
            0,
            dst.unsqueeze(1).expand(-1, n_src),
            norm_ew.unsqueeze(1) * signals[src]              # (E, n_src)
        )
        signals = torch.clamp(signals + msg, 0., 1.)

    received = signals[source_nodes]                         # (n_src, n_src)
    off_diag = 1. - torch.eye(n_src, device=DEVICE)
    loss     = F.relu(1. - received) * off_diag              # 미도달 패널티

    return loss.sum() * weight


# ── 손실 함수 ─────────────────────────────────────────────────────────────────
def loss_fn(ew: torch.Tensor, gs: torch.Tensor, c: dict, scale: float) -> torch.Tensor:
    src, dst = EDGE_INDEX[0], EDGE_INDEX[1]
    ridge    = c['ridge']
    gt_tgt   = c['gt_edge_target']            # 연속 통행량 타깃 (0~1) 또는 None

    # 1. Ridge: 자유공간 중심선을 따라 도로 유도.
    #    단, 통행량 GT가 있는 엣지(gt_tgt>0)는 면제 — GT 경로가 skeleton을
    #    벗어나 곧게 가도 벌점을 받지 않게 ("구비구비" 완화)
    rs         = (ridge[src] + ridge[dst]) * 0.5
    ridge_pen  = ew * (1. - rs)
    if gt_tgt is not None:
        ridge_pen = ridge_pen * (1. - (gt_tgt > 0).float())
    ridge_loss = ridge_pen.sum() * (scale * 8. + 0.3)

    # 2. Connectivity: 건물 + 관문 노드 간 실제 경로 존재 강제 (확산 방식)
    #    관문 선택은 detach — 위치 선택 자체에는 gradient를 흘리지 않고,
    #    선택된 관문까지 도로(ew)가 자라도록만 학습
    gate_nodes = torch.tensor(
        [BOUNDARY[i] for i in _pick_gates(gs.detach())],
        dtype=torch.long, device=DEVICE)
    gate_nodes = gate_nodes[c['is_building'][gate_nodes] < 1]   # 건물 위 관문 제외
    srcs = torch.cat([c['building_nodes'], gate_nodes])
    conn = diffusion_conn_loss(ew, srcs, EDGE_INDEX, T=DIFF_STEPS,
                               weight=scale * 25. + 1.)

    # 3. Sparsity: 총 도로량 억제 (가느다란 길 유도)
    sparse = ew.mean() * SPARSE_WEIGHT

    # 4. Degree: 과도한 교차점 억제
    strength = torch.zeros(N, device=DEVICE)
    strength.scatter_add_(0, src, ew)
    strength.scatter_add_(0, dst, ew)
    degree = F.relu(strength - DEGREE_THRESH).sum() * DEGREE_WEIGHT

    # 5. Gate: 정확히 N_GATES개 관문 활성화
    gate = (gs.sum() - N_GATES).pow(2) * 300.

    # 6. GT road: 통행량 위계 타깃(연속 0~1)으로 ew를 soft-target BCE 유도.
    #    줄기(타깃≈1)는 강하게, 잔가지(타깃≈0.1)는 약하게 → ew가 위계를 학습.
    #    타깃이 0인 엣지(경로 미통과)는 sparse가 이미 0으로 누르므로 제외.
    if gt_tgt is not None:
        m = gt_tgt > 0
        gt_loss = F.binary_cross_entropy(
            ew[m].clamp(1e-6, 1. - 1e-6),
            gt_tgt[m]
        ) * (scale * GT_LOSS_WEIGHT + 5.)
    else:
        gt_loss = ew.sum() * 0.

    return ridge_loss + conn + sparse + degree + gate + gt_loss


# ── 학습 ─────────────────────────────────────────────────────────────────────
def train(campuses: list[dict], epochs: int = EPOCHS) -> MultiCampusRoadGAT:
    model = MultiCampusRoadGAT().to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters())
    print(f'  파라미터: {n_params:,}개 (campus-agnostic 공유)')

    opt = optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    sch = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=5e-6)

    stopped = False
    def _sig(s, f):
        nonlocal stopped
        print('\n⚠️  중단 감지 — 저장 후 종료...')
        stopped = True
    signal.signal(signal.SIGINT, _sig)

    n = len(campuses)
    print(f'▶ 학습: {n}개 캠퍼스 × {epochs} epochs  (확산 스텝={DIFF_STEPS})  |  device={DEVICE}')

    for ep in range(1, epochs + 1):
        if stopped:
            break
        model.train()
        scale = min(1.0, ep / WARMUP_EPOCHS)
        total = 0.

        # 전체 캠퍼스 gradient 누적 후 한 번만 step (GPU 효율 ↑)
        opt.zero_grad()
        for c in campuses:
            ew, gs = model(c['node_feats'], EDGE_INDEX, c['bld_mask'])
            loss   = loss_fn(ew, gs, c, scale) / n   # 캠퍼스 수로 정규화
            loss.backward()
            total += loss.item() * n
        torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
        opt.step()
        sch.step()

        log_every = max(1, epochs // 10)
        if ep % log_every == 0 or ep == 1:
            print(f'  [{ep:4d}/{epochs}]  avg_loss={total/n:10.2f}'
                  f'  lr={sch.get_last_lr()[0]:.2e}')

    torch.save(model.state_dict(), CKPT)
    print(f'\n✅ 모델 저장: {CKPT}')
    return model


# ── 도로망 추출 (최단경로 기반) ───────────────────────────────────────────────
def _extract_roads(ew_np: np.ndarray, terminals: list[int],
                   ib_np: np.ndarray) -> dict[tuple[int, int], int]:
    """학습된 엣지 가중치를 비용(1-ew)으로 터미널(건물+관문) 간 도로망 추출.

    터미널 완전그래프의 최단거리 MST에 해당하는 경로들을 합쳐 도로망을 만들고,
    여러 경로가 공유하는 구간은 사용 횟수를 누적한다(간선별 굵기에 사용).
    상위 percentile 엣지를 점으로 찍는 방식과 달리 항상 연결된 길이 나오고,
    모든 건물이 도로망에 포함된다.
    """
    src = EDGE_INDEX[0].cpu().numpy()
    dst = EDGE_INDEX[1].cpu().numpy()
    valid = (src != dst) & (ib_np[src] == 0) & (ib_np[dst] == 0)
    s, d  = src[valid], dst[valid]
    cost  = 1.001 - ew_np[valid]                       # ew↑ → 비용↓ (항상 양수)
    graph = csr_matrix((cost, (s, d)), shape=(N, N))

    terms = np.array(sorted(set(terminals)), dtype=np.int64)
    if len(terms) < 2:
        return {}
    dmat, pred = dijkstra(graph, indices=terms, return_predecessors=True)

    # 터미널 간 거리 행렬 → MST
    tmat = dmat[:, terms]
    tmat[~np.isfinite(tmat)] = 1e9                     # 도달 불가(건물에 갇힌 노드) 쌍
    mst  = minimum_spanning_tree(csr_matrix(tmat))

    seg_count: dict[tuple[int, int], int] = {}
    for a, b in zip(*mst.nonzero()):
        node = int(terms[b])                           # 터미널 b → a 로 역추적
        while node != terms[a]:
            p = int(pred[a, node])
            if p < 0:                                  # 경로 없음
                break
            key = (min(p, node), max(p, node))
            seg_count[key] = seg_count.get(key, 0) + 1
            node = p
    return seg_count


# ── 추론 & 시각화 ─────────────────────────────────────────────────────────────
def plot_campus(model: MultiCampusRoadGAT, c: dict, out_path: str):
    model.eval()
    with torch.no_grad():
        ew, gs = model(c['node_feats'], EDGE_INDEX, c['bld_mask'])
    ew_np = ew.cpu().numpy()
    ib_np = c['is_building'].cpu().numpy()
    W, H  = c['W'], c['H']

    chosen     = _pick_gates(gs)
    gate_nodes = [BOUNDARY[i] for i in chosen if ib_np[BOUNDARY[i]] == 0]
    terminals  = c['building_nodes'].cpu().tolist() + gate_nodes
    segs       = _extract_roads(ew_np, terminals, ib_np)

    fig, ax = plt.subplots(figsize=(12, 12))
    ax.set_facecolor('#0d1117')
    fig.patch.set_facecolor('#0d1117')

    # 건물 폴리곤
    for bid, pts in c['poly'].items():
        scaled = [(p[0] * RES / W, p[1] * RES / H) for p in pts]
        ax.add_patch(mpatches.Polygon(
            scaled, closed=True,
            facecolor='#2d3436', edgecolor='#00cec9', lw=0.8, alpha=0.9))
        cx = np.mean([p[0] for p in scaled])
        cy = np.mean([p[1] for p in scaled])
        ax.text(cx, cy, bid, color='white', ha='center', va='center',
                fontsize=5.5, fontweight='bold')

    # 도로망: 공유 구간(간선 사용 횟수)이 많을수록 굵게 → 간선도로/지선 위계
    if segs:
        mx_cnt = max(segs.values())
        for (s_, d_), cnt in segs.items():
            y1, x1 = divmod(s_, RES)
            y2, x2 = divmod(d_, RES)
            lw = 1.5 + 4.5 * (cnt / mx_cnt)
            ax.plot([x1, x2], [y1, y2], color='#ffd32a', alpha=0.9,
                    lw=lw, solid_capstyle='round')

    # 관문
    labels = ['GATE A', 'GATE B', 'GATE C', 'GATE D']
    for k, gi in enumerate(chosen):
        node = BOUNDARY[gi]
        gx, gy = node % RES, node // RES
        ax.scatter(gx, gy, s=500, c='#ff3838', marker='*',
                   edgecolors='white', lw=1.5, zorder=10)
        ax.text(gx, gy - 2.5, labels[k], color='white', ha='center',
                fontsize=9, fontweight='bold',
                bbox=dict(fc='#d63031', alpha=0.8, ec='none', pad=1))

    ax.set_xlim(0, RES)
    ax.set_ylim(RES, 0)
    ax.axis('off')
    title = c['name'].replace('_', ' ').title()
    ax.set_title(f'RWS_V11  ·  Multi-Campus GAT V23 (통행량 위계 GT)  ·  {title}',
                 color='white', fontsize=13, pad=10)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight', facecolor='#0d1117')
    plt.close()
    print(f'  → {os.path.basename(out_path)}')


def visualize_all(model: MultiCampusRoadGAT, campuses: list[dict], out_dir: str = OUT_DIR):
    print(f'\n▶ {len(campuses)}개 캠퍼스 시각화 → {out_dir}')
    os.makedirs(out_dir, exist_ok=True)
    for c in campuses:
        plot_campus(model, c, os.path.join(out_dir, f'RWS_V11_road_{c["name"]}.png'))


# ── 메인 ─────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--load',  action='store_true',
                        help='저장된 체크포인트 불러오기')
    parser.add_argument('--infer', type=str, default=None,
                        help='특정 캠퍼스만 추론 (예: sungkyunkwan_university)')
    parser.add_argument('--epochs', type=int, default=EPOCHS,
                        help=f'학습 epoch 수 (기본 {EPOCHS})')
    args = parser.parse_args()

    campuses = load_all()
    train_campuses = [c for c in campuses if c['name'] not in TEST_CAMPUSES]
    test_campuses  = [c for c in campuses if c['name'] in TEST_CAMPUSES]
    print(f'  Train: {len(train_campuses)}개  /  Test(hold-out): {len(test_campuses)}개')

    model = MultiCampusRoadGAT().to(DEVICE)

    use_ckpt = (args.load or args.infer is not None) and os.path.exists(CKPT)
    if use_ckpt:
        model.load_state_dict(torch.load(CKPT, map_location=DEVICE))
        print(f'✓ 체크포인트 로드: {CKPT}\n')
    else:
        if args.load:
            print(f'[WARN] 체크포인트 없음 ({CKPT}) → 새로 학습\n')
        model = train(train_campuses, epochs=args.epochs)

    if args.infer:
        hits = [c for c in campuses if args.infer.lower() in c['name'].lower()]
        if not hits:
            print(f'[ERROR] "{args.infer}" 매칭 캠퍼스 없음')
            print('  사용 가능:', [c['name'] for c in campuses])
        else:
            plot_campus(model, hits[0],
                        os.path.join(OUT_DIR, f'RWS_V11_road_{hits[0]["name"]}.png'))
    else:
        visualize_all(model, test_campuses, out_dir=os.path.join(OUT_DIR, 'gat_test'))
