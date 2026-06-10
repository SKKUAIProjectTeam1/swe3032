"""
multi_campus_gat_train.py  —  Multi-Campus Dataset GAT Road Designer (train/test 버전)

gat2.py 와의 핵심 차이:
  ✗ 단일 캠퍼스 1장으로 학습, campus-specific node_embed 암기
  ✓ 20개 대학 이미지를 데이터셋으로 학습, 가중치 완전 공유
  ✓ 어떤 캠퍼스에도 적용 가능한 campus-agnostic GAT

실행:
  python model/multi_campus_gat_train.py                      # 학습(hold-out 제외) → hold-out 시각화
  python model/multi_campus_gat_train.py --epochs 20          # epoch 수 조절 (시험 실행)
  python model/multi_campus_gat_train.py --load               # 저장 모델 로드 → hold-out 시각화
  python model/multi_campus_gat_train.py --load --infer sungkyunkwan_university

TEST_CAMPUSES(5개)는 학습에서 제외되는 hold-out 캠퍼스로, V21 size_test와 동일한
캠퍼스를 사용해 두 모델의 결과를 직접 비교할 수 있다.
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

sys.stdout.reconfigure(encoding='utf-8')

# ── 설정 ─────────────────────────────────────────────────────────────────────
RES           = 100          # 격자 해상도 (RES × RES 노드)
N             = RES * RES
EPOCHS        = 300
LR            = 3e-4
WARMUP_EPOCHS = 50
N_GATES       = 2            # 캠퍼스당 추천 관문 수
GATE_MIN_DIST = 14           # 관문 간 최소 L1 거리 (격자 단위)
DEVICE        = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

ROOT    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
IMG_DIR = os.path.join(ROOT, 'collegemap', 'images')
TXT_DIR = os.path.join(ROOT, 'collegemap', 'txt')
OUT_DIR = os.path.join(ROOT, 'output')
CKPT    = os.path.join(OUT_DIR, 'multi_campus_gat.pth')
os.makedirs(OUT_DIR, exist_ok=True)

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


def load_campus(img_path: str, txt_path: str) -> dict:
    # 건물 마스크 이미지 → RES×RES
    img    = Image.open(img_path).convert('L')
    W, H   = img.size
    small  = img.resize((RES, RES), resample=Image.NEAREST)
    is_bld = (np.array(small) > 128).astype(np.float32)   # (RES, RES)

    # ridge: 건물에서 1.5~7 격자 거리 → 건물 사이 통로 후보
    dist  = distance_transform_edt(1 - is_bld)
    ridge = ((dist > 1.5) & (dist < 7.0)).astype(np.float32)

    # 노드 피처 4채널: [is_building, ridge, x_norm, y_norm]
    xs = np.tile(np.arange(RES),          RES).astype(np.float32) / RES
    ys = np.repeat(np.arange(RES), RES       ).astype(np.float32) / RES
    node_feats = np.stack(
        [is_bld.flatten(), ridge.flatten(), xs, ys], axis=1
    ).astype(np.float32)                                   # (N, 4)

    # BUILDING_POLY 로드 → 건물 중심 노드 인덱스 (RES×RES 좌표)
    ns = {}
    with open(txt_path, encoding='utf-8') as f:
        exec(f.read(), ns)
    poly = ns['BUILDING_POLY']

    bld_nodes = list({
        max(0, min(N - 1,
                   int(np.mean([p[1] for p in pts]) * RES / H) * RES
                   + int(np.mean([p[0] for p in pts]) * RES / W)))
        for pts in poly.values()
    })

    name = os.path.basename(img_path).replace('_building_mask.png', '')
    return {
        'node_feats':    torch.tensor(node_feats,       dtype=torch.float32).to(DEVICE),
        'is_building':   torch.tensor(is_bld.flatten(), dtype=torch.float32).to(DEVICE),
        'ridge':         torch.tensor(ridge.flatten(),  dtype=torch.float32).to(DEVICE),
        'building_nodes': torch.tensor(bld_nodes,       dtype=torch.long   ).to(DEVICE),
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


# ── Sparse GAT 레이어 ─────────────────────────────────────────────────────────
class SparseGATLayer(nn.Module):
    """
    h'[src] = LayerNorm(ELU(Σ_{dst} α_{src→dst} · W·h[dst]))
    α: LeakyReLU attention, normalized per source node
    """
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.W        = nn.Linear(in_dim, out_dim, bias=False)
        self.attn_src = nn.Linear(out_dim, 1,      bias=False)
        self.attn_dst = nn.Linear(out_dim, 1,      bias=False)
        self.norm     = nn.LayerNorm(out_dim)
        for lin in (self.W, self.attn_src, self.attn_dst):
            nn.init.xavier_uniform_(lin.weight, gain=0.5)

    def forward(self, x: torch.Tensor, ei: torch.Tensor) -> torch.Tensor:
        src, dst = ei[0], ei[1]
        h = self.W(x)                                              # (N, out)
        e = F.leaky_relu(
            self.attn_src(h)[src].squeeze(-1) +
            self.attn_dst(h)[dst].squeeze(-1), 0.2)               # (E,)
        e_exp = torch.exp(e - e.max())
        denom = torch.zeros(N, device=x.device).scatter_add_(0, src, e_exp)
        alpha = e_exp / (denom[src] + 1e-8)                        # (E,)
        agg   = torch.zeros(N, h.shape[1], device=x.device)
        agg.scatter_add_(0,
                         src.unsqueeze(1).expand(-1, h.shape[1]),
                         alpha.unsqueeze(1) * h[dst])
        return self.norm(F.elu(agg))


# ── MultiCampusRoadGAT ────────────────────────────────────────────────────────
class MultiCampusRoadGAT(nn.Module):
    """
    Campus-agnostic: 어떤 캠퍼스든 동일 가중치로 처리.
    입력: node_feats (N, 4)  [is_building, ridge, x_norm, y_norm]
    출력: edge_weights (E,), gate_scores (|BOUNDARY|,)
    """
    def __init__(self):
        super().__init__()
        self.gat1 = SparseGATLayer(4,  64)
        self.gat2 = SparseGATLayer(64, 32)
        self.gat3 = SparseGATLayer(32, 16)

        # edge_head: cat(h2[src], h2[dst]) = (E, 64) → 도로 확률
        self.edge_head = nn.Sequential(
            nn.Linear(64, 16), nn.ReLU(),
            nn.Linear(16,  1),
        )
        # gate_head: h3[boundary] = (|B|, 16) → 관문 점수
        self.gate_head = nn.Linear(16, 1)

        for m in self.modules():
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.zeros_(m.bias)

    def forward(self, feats: torch.Tensor, ei: torch.Tensor):
        h1 = self.gat1(feats, ei)                                   # (N, 64)
        h2 = self.gat2(h1,    ei)                                   # (N, 32)
        h3 = self.gat3(h2,    ei)                                   # (N, 16)
        src, dst = ei[0], ei[1]
        ew = torch.sigmoid(
            self.edge_head(torch.cat([h2[src], h2[dst]], dim=1)).squeeze(-1)
        )                                                            # (E,)
        gs = torch.sigmoid(
            self.gate_head(h3[BOUNDARY_T]).squeeze(-1)
        )                                                            # (|B|,)
        return ew, gs


# ── 손실 함수 ─────────────────────────────────────────────────────────────────
def loss_fn(ew: torch.Tensor, gs: torch.Tensor, c: dict, scale: float) -> torch.Tensor:
    src, dst = EDGE_INDEX[0], EDGE_INDEX[1]
    ib, ridge, bnds = c['is_building'], c['ridge'], c['building_nodes']

    # 1. Collision: 건물 관통 도로 억제 (warmup으로 점진적 강화)
    col_mask  = (ib[src] > 0) | (ib[dst] > 0)
    collision = ew[col_mask].sum() * (scale * 5000.)

    # 2. Ridge: 건물 사이 통로를 따라 도로 유도
    rs    = (ridge[src] + ridge[dst]) * 0.5
    ridge_loss = (ew * (1. - rs)).sum() * (scale * 8. + 0.3)

    # 3. Connectivity: 모든 건물 노드가 도로에 연결되어야 함
    strength = torch.zeros(N, device=DEVICE)
    strength.scatter_add_(0, src, ew)
    strength.scatter_add_(0, dst, ew)
    conn = F.relu(2. - strength[bnds]).pow(2).sum() * (scale * 25. + 1.)

    # 4. Sparsity: 총 도로량 억제 (가느다란 길 유도)
    sparse = ew.mean() * 15.

    # 5. Degree: 과도한 교차점 억제
    degree = F.relu(strength - 4.).sum() * 3.

    # 6. Gate: 정확히 N_GATES개 관문 활성화
    gate = (gs.sum() - N_GATES).pow(2) * 300.

    return collision + ridge_loss + conn + sparse + degree + gate


# ── 학습 ─────────────────────────────────────────────────────────────────────
def train(campuses: list[dict], epochs: int = EPOCHS) -> MultiCampusRoadGAT:
    model = MultiCampusRoadGAT().to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters())
    print(f'  파라미터: {n_params:,}개 (campus-agnostic 공유)')

    opt = optim.Adam(model.parameters(), lr=LR)
    sch = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=5e-6)

    stopped = False
    def _sig(s, f):
        nonlocal stopped
        print('\n⚠️  중단 감지 — 저장 후 종료...')
        stopped = True
    signal.signal(signal.SIGINT, _sig)

    n = len(campuses)
    print(f'▶ 학습: {n}개 캠퍼스 × {epochs} epochs  |  device={DEVICE}')

    # torch.compile: CUDA 있을 때만 (첫 epoch 느리지만 이후 빠름)
    compiled = model
    if DEVICE.type == 'cuda':
        try:
            compiled = torch.compile(model)
            print('  torch.compile 활성화 (첫 epoch 워밍업)')
        except Exception:
            pass

    for ep in range(1, epochs + 1):
        if stopped:
            break
        compiled.train()
        scale = min(1.0, ep / WARMUP_EPOCHS)
        total = 0.

        # 전체 캠퍼스 gradient 누적 후 한 번만 step (GPU 효율 ↑)
        opt.zero_grad()
        for c in campuses:
            ew, gs = compiled(c['node_feats'], EDGE_INDEX)
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


# ── 추론 & 시각화 ─────────────────────────────────────────────────────────────
def plot_campus(model: MultiCampusRoadGAT, c: dict, out_path: str):
    model.eval()
    with torch.no_grad():
        ew, gs = model(c['node_feats'], EDGE_INDEX)
    ew_np = ew.cpu().numpy()
    gs_np = gs.cpu().numpy()
    ei    = EDGE_INDEX.cpu()
    ib_np = c['is_building'].cpu().numpy()
    W, H  = c['W'], c['H']

    # 관문 greedy 선택 (거리 다양성 보장)
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

    # 도로 엣지 (상위 1.5% 만 표시)
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
    ax.set_title(f'Multi-Campus GAT  ·  {title}',
                 color='white', fontsize=13, pad=10)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight', facecolor='#0d1117')
    plt.close()
    print(f'  → {os.path.basename(out_path)}')


def visualize_all(model: MultiCampusRoadGAT, campuses: list[dict], out_dir: str = OUT_DIR):
    print(f'\n▶ {len(campuses)}개 캠퍼스 시각화 → {out_dir}')
    os.makedirs(out_dir, exist_ok=True)
    for c in campuses:
        plot_campus(model, c, os.path.join(out_dir, f'road_{c["name"]}.png'))


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
                        os.path.join(OUT_DIR, f'road_{hits[0]["name"]}.png'))
    else:
        visualize_all(model, test_campuses, out_dir=os.path.join(OUT_DIR, 'gat_test'))
