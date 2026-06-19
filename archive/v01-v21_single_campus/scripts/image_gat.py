"""
image_gat.py — Multi-Campus GAT Road Designer  (V22-B)
────────────────────────────────────────────────────────
[Option B] 그래프 확산 기반 연결성 손실
  - 각 건물 인접 노드에서 신호를 쏘고 T번 전파
  - T 스텝 후 다른 건물 인접 노드에 신호가 도달해야 함
  - 외곽만 빙 돌면 건물 간 실제 경로가 없어 패널티

[버그 수정]
  - hard masking + 건물 중심 노드 사용 시 connectivity loss가 0이 되는 문제
  - building_nodes를 각 건물의 가장 가까운 비건물 노드로 변경

실행:
  python model/image_gat.py               # 학습 → 전체 시각화
  python model/image_gat.py --load        # 저장 모델 로드 → 시각화
  python model/image_gat.py --load --infer sungkyunkwan_university
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
from scipy.ndimage import distance_transform_edt

# ── 설정 ─────────────────────────────────────────────────────────────────────
RES           = 100
N             = RES * RES
NODE_DIM      = 5           # [is_building, ridge, dist_norm, x_norm, y_norm]
GAT_HEADS     = 4
DIFF_STEPS    = 25          # 확산 스텝 수 (건물 간 최대 거리 커버)
EPOCHS        = 300
LR            = 3e-4
WARMUP_EPOCHS = 50
ATTN_DROPOUT  = 0.1
N_GATES       = 2
GATE_MIN_DIST = 14
DEVICE        = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

ROOT    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
IMG_DIR = os.path.join(ROOT, 'collegemap', 'images')
TXT_DIR = os.path.join(ROOT, 'collegemap', 'txt')
OUT_DIR = os.path.join(ROOT, 'output')
CKPT    = os.path.join(OUT_DIR, 'image_gat.pth')
os.makedirs(OUT_DIR, exist_ok=True)


# ── 고정 그래프 토폴로지 ──────────────────────────────────────────────────────
def _build_edges(res: int) -> torch.Tensor:
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

_bdy = []
for i in range(10, RES - 10):
    _bdy += [i, (RES - 1) * RES + i, i * RES, i * RES + (RES - 1)]
BOUNDARY   = sorted(set(_bdy))
BOUNDARY_T = torch.tensor(BOUNDARY, dtype=torch.long).to(DEVICE)


# ── 데이터 로딩 ────────────────────────────────────────────────────────────────
def _find_txt(img_path: str) -> str | None:
    stem   = os.path.basename(img_path).replace('_building_mask.png', '')
    direct = os.path.join(TXT_DIR, stem + '_building_places.txt')
    if os.path.exists(direct):
        return direct
    prefix = stem.replace('-', '_').split('_')[0]
    for fn in os.listdir(TXT_DIR):
        if fn.endswith('_building_places.txt') and fn.startswith(prefix):
            return os.path.join(TXT_DIR, fn)
    return None


def _nearest_road_node(cy: float, cx: float, is_bld_grid: np.ndarray) -> int:
    """건물 중심(cy, cx)에서 가장 가까운 비건물 노드 인덱스 반환"""
    non_bld_yx = np.argwhere(is_bld_grid == 0)   # (K, 2) [y, x]
    if len(non_bld_yx) == 0:
        return int(cy) * RES + int(cx)
    dists = (non_bld_yx[:, 0] - cy) ** 2 + (non_bld_yx[:, 1] - cx) ** 2
    best  = non_bld_yx[np.argmin(dists)]
    return int(best[0]) * RES + int(best[1])


def load_campus(img_path: str, txt_path: str) -> dict:
    img    = Image.open(img_path).convert('L')
    W, H   = img.size
    small  = img.resize((RES, RES), resample=Image.NEAREST)
    is_bld = (np.array(small) > 128).astype(np.float32)   # (RES, RES)

    dist   = distance_transform_edt(1 - is_bld)
    ridge  = ((dist > 1.5) & (dist < 7.0)).astype(np.float32)
    d_max  = dist.max() if dist.max() > 0 else 1.0
    dist_n = (dist / d_max).astype(np.float32)

    xs = np.tile(np.arange(RES), RES).astype(np.float32) / RES
    ys = np.repeat(np.arange(RES), RES).astype(np.float32) / RES
    node_feats = np.stack(
        [is_bld.flatten(), ridge.flatten(), dist_n.flatten(), xs, ys], axis=1
    ).astype(np.float32)

    ib_flat  = torch.tensor(is_bld.flatten(), dtype=torch.float32).to(DEVICE)
    src, dst = EDGE_INDEX[0], EDGE_INDEX[1]
    bld_mask = (ib_flat[src] > 0) | (ib_flat[dst] > 0)

    ns = {}
    with open(txt_path, encoding='utf-8') as f:
        exec(f.read(), ns)
    poly = ns['BUILDING_POLY']

    # building_nodes: 각 건물 중심에서 가장 가까운 비건물 노드
    bld_nodes = list({
        _nearest_road_node(
            np.mean([p[1] for p in pts]) * RES / H,
            np.mean([p[0] for p in pts]) * RES / W,
            is_bld
        )
        for pts in poly.values()
    })

    name = os.path.basename(img_path).replace('_building_mask.png', '')
    return {
        'node_feats':     torch.tensor(node_feats,       dtype=torch.float32).to(DEVICE),
        'is_building':    ib_flat,
        'bld_mask':       bld_mask,
        'ridge':          torch.tensor(ridge.flatten(),  dtype=torch.float32).to(DEVICE),
        'building_nodes': torch.tensor(bld_nodes,        dtype=torch.long   ).to(DEVICE),
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
            print(f'  ✓ {c["name"]:60s} ({len(c["building_nodes"])} 건물)')
        except Exception as e:
            print(f'  ✗ {os.path.basename(img_path)}: {e}')
    print(f'  → 총 {len(campuses)}개 로드 완료\n')
    return campuses


# ── Multi-Head Sparse GAT Layer ───────────────────────────────────────────────
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


# ── Main Model ────────────────────────────────────────────────────────────────
class CampusGAT(nn.Module):
    def __init__(self):
        super().__init__()
        self.gat1      = MHGATLayer(NODE_DIM, 64, heads=GAT_HEADS, dropout=ATTN_DROPOUT)
        self.gat2      = MHGATLayer(64,       64, heads=GAT_HEADS, dropout=ATTN_DROPOUT)
        self.gat3      = MHGATLayer(64,       32, heads=GAT_HEADS, dropout=ATTN_DROPOUT)
        self.edge_head = nn.Sequential(
            nn.Linear(128, 32), nn.ReLU(), nn.Dropout(0.1), nn.Linear(32, 1)
        )
        self.gate_head = nn.Linear(32, 1)
        for m in self.modules():
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.zeros_(m.bias)
        # sparse 초기화: sigmoid(-2) ≈ 0.12 → 초기 ridge_loss 최소화
        nn.init.constant_(self.edge_head[-1].bias, -2.0)

    def forward(self, feats: torch.Tensor, ei: torch.Tensor, bld_mask: torch.Tensor):
        h1  = self.gat1(feats, ei)
        h2  = self.gat2(h1,   ei)
        h3  = self.gat3(h2,   ei)
        src, dst = ei[0], ei[1]
        raw = self.edge_head(torch.cat([h2[src], h2[dst]], dim=1)).squeeze(-1)
        ew  = torch.sigmoid(raw) * (~bld_mask).float()   # hard masking
        gs  = torch.sigmoid(self.gate_head(h3[BOUNDARY_T]).squeeze(-1))
        return ew, gs


# ── [Option B] 그래프 확산 연결성 손실 ────────────────────────────────────────
def diffusion_conn_loss(
    ew:             torch.Tensor,  # (E,)
    building_nodes: torch.Tensor,  # (B,)
    ei:             torch.Tensor,  # (2,E)
    T:              int   = DIFF_STEPS,
    weight:         float = 1.0,
) -> torch.Tensor:
    """
    각 건물 인접 노드에서 신호를 발사해 T번 전파.
    - row-normalize 제거: 신호 소멸 문제 해결
    - steep sigmoid: 신호가 임계값(0.3) 넘으면 1로 스냅 → 장거리 전파 가능
    """
    src, dst = ei[0], ei[1]
    n_bld    = building_nodes.size(0)
    if n_bld <= 1:
        return ew.sum() * 0.

    signals = torch.zeros(N, n_bld, device=DEVICE)
    signals[building_nodes, torch.arange(n_bld, device=DEVICE)] = 1.0

    for _ in range(T):
        msg = torch.zeros(N, n_bld, device=DEVICE)
        msg.scatter_add_(
            0,
            dst.unsqueeze(1).expand(-1, n_bld),
            ew.unsqueeze(1) * signals[src]
        )
        # steep sigmoid: 0.3 초과하면 빠르게 1로 수렴 → 소멸 없이 장거리 전파
        signals = torch.sigmoid(10. * (signals + msg - 0.3))

    received = signals[building_nodes]                        # (n_bld, n_bld)
    off_diag = 1. - torch.eye(n_bld, device=DEVICE)
    return F.relu(1. - received).mul(off_diag).sum() * weight


# ── 손실 함수 ─────────────────────────────────────────────────────────────────
def loss_fn(ew: torch.Tensor, gs: torch.Tensor, c: dict, scale: float) -> torch.Tensor:
    src, dst = EDGE_INDEX[0], EDGE_INDEX[1]
    rs = (c['ridge'][src] + c['ridge'][dst]) * 0.5

    # Ridge: sum→mean 으로 스케일 정상화 (기존 대비 ~80,000배 감소)
    ridge_loss = (ew * (1. - rs)).mean() * (scale * 3. + 0.1)

    # Connectivity: 확산 (sigmoid 전파로 장거리 동작)
    conn = diffusion_conn_loss(ew, c['building_nodes'], EDGE_INDEX,
                               T=DIFF_STEPS, weight=scale * 20. + 1.)

    # Sparsity
    sparse = ew.mean() * 3.

    # Degree: sum→mean (N=10,000으로 나눔)
    strength = torch.zeros(N, device=DEVICE)
    strength.scatter_add_(0, src, ew)
    strength.scatter_add_(0, dst, ew)
    degree = F.relu(strength - 4.).mean() * 300.

    # Gate
    gate = (gs.sum() - N_GATES).pow(2) * 300.

    return ridge_loss + conn + sparse + degree + gate


# ── 학습 ─────────────────────────────────────────────────────────────────────
def train(campuses: list[dict]) -> CampusGAT:
    model   = CampusGAT().to(DEVICE)
    n_param = sum(p.numel() for p in model.parameters())
    print(f'  파라미터: {n_param:,}개  |  device={DEVICE}')

    opt = optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    sch = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS, eta_min=5e-6)

    stopped = False
    def _sig(s, f):
        nonlocal stopped
        print('\n⚠️  중단 감지 — 저장 후 종료...')
        stopped = True
    signal.signal(signal.SIGINT, _sig)

    n = len(campuses)
    print(f'▶ 학습: {n}개 캠퍼스 × {EPOCHS} epochs  (확산 스텝={DIFF_STEPS})')

    for ep in range(1, EPOCHS + 1):
        if stopped:
            break
        model.train()
        scale = min(1.0, ep / WARMUP_EPOCHS)
        total = 0.

        opt.zero_grad()
        for c in campuses:
            ew, gs = model(c['node_feats'], EDGE_INDEX, c['bld_mask'])
            loss   = loss_fn(ew, gs, c, scale) / n
            loss.backward()
            total += loss.item() * n
        torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
        opt.step()
        sch.step()

        if ep % 25 == 0 or ep == 1:
            print(f'  [{ep:4d}/{EPOCHS}]  avg_loss={total/n:10.2f}'
                  f'  lr={sch.get_last_lr()[0]:.2e}')

    torch.save(model.state_dict(), CKPT)
    print(f'\n✅ 모델 저장: {CKPT}')
    return model


# ── 추론 & 시각화 ─────────────────────────────────────────────────────────────
def plot_campus(model: CampusGAT, c: dict, out_path: str):
    model.eval()
    with torch.no_grad():
        ew, gs = model(c['node_feats'], EDGE_INDEX, c['bld_mask'])
    ew_np = ew.cpu().numpy()
    gs_np = gs.cpu().numpy()
    ei    = EDGE_INDEX.cpu()
    ib_np = c['is_building'].cpu().numpy()
    W, H  = c['W'], c['H']

    order  = np.argsort(gs_np)[::-1]
    chosen = []
    for gi in order:
        gx = BOUNDARY[gi] % RES
        gy = BOUNDARY[gi] // RES
        if all(abs(gx - BOUNDARY[p] % RES) + abs(gy - BOUNDARY[p] // RES) >= GATE_MIN_DIST
               for p in chosen):
            chosen.append(gi)
        if len(chosen) == N_GATES:
            break

    fig, ax = plt.subplots(figsize=(12, 12))
    ax.set_facecolor('#0d1117')
    fig.patch.set_facecolor('#0d1117')

    for bid, pts in c['poly'].items():
        scaled = [(p[0] * RES / W, p[1] * RES / H) for p in pts]
        ax.add_patch(mpatches.Polygon(
            scaled, closed=True,
            facecolor='#2d3436', edgecolor='#00cec9', lw=0.8, alpha=0.9))
        cx = np.mean([p[0] for p in scaled])
        cy = np.mean([p[1] for p in scaled])
        ax.text(cx, cy, bid, color='white', ha='center', va='center',
                fontsize=5.5, fontweight='bold')

    thr = np.percentile(ew_np, 98.5)
    mxw = max(ew_np.max(), thr + 1e-8)
    for i in range(len(ew_np)):
        w = ew_np[i]
        if w < thr:
            continue
        s, d = ei[0, i].item(), ei[1, i].item()
        if ib_np[s] > 0 or ib_np[d] > 0:
            continue
        y1, x1 = divmod(s, RES)
        y2, x2 = divmod(d, RES)
        lw = (w - thr) / (mxw - thr) * 8 + 1
        ax.plot([x1, x2], [y1, y2], color='#ffd32a', alpha=0.85,
                lw=lw, solid_capstyle='round')

    labels = ['GATE A', 'GATE B']
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
    ax.set_title(f'Campus GAT V22-B (Diffusion Conn)  ·  {title}',
                 color='white', fontsize=13, pad=10)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight', facecolor='#0d1117')
    plt.close()
    print(f'  → {os.path.basename(out_path)}')


def visualize_all(model: CampusGAT, campuses: list[dict]):
    print(f'\n▶ 전체 {len(campuses)}개 캠퍼스 시각화...')
    for c in campuses:
        plot_campus(model, c, os.path.join(OUT_DIR, f'v22b_{c["name"]}.png'))


# ── 메인 ─────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--load',  action='store_true')
    parser.add_argument('--infer', type=str, default=None)
    args = parser.parse_args()

    campuses = load_all()
    model    = CampusGAT().to(DEVICE)

    use_ckpt = (args.load or args.infer is not None) and os.path.exists(CKPT)
    if use_ckpt:
        model.load_state_dict(torch.load(CKPT, map_location=DEVICE))
        print(f'✓ 체크포인트 로드: {CKPT}\n')
    else:
        if args.load:
            print(f'⚠️  체크포인트 없음 ({CKPT}), 새로 학습 시작')
        model = train(campuses)

    if args.infer:
        target = [c for c in campuses if args.infer in c['name']]
        if not target:
            print(f'[ERROR] 캠퍼스 없음: {args.infer}')
        else:
            plot_campus(model, target[0],
                        os.path.join(OUT_DIR, f'v22b_{target[0]["name"]}.png'))
    else:
        visualize_all(model, campuses)
