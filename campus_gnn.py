"""
campus_gnn.py
건물 간 이동 그래프 위에서 다음 슬롯 체류 인원 예측.

학습: 2025-1학기 synthetic augmented data (200 instances)
테스트: 2025-2학기 실제 snapshot (cross-semester evaluation)

비교:
  MLP  — 그래프 구조 무시, 각 건물 독립적으로 예측
  GCN  — 이웃 건물 정보를 집계해 예측
  GAT  — 이웃마다 attention weight 학습

실행:
  python campus_gnn.py
  python campus_gnn.py --eval-day 목
"""
import sys, argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import koreanize_matplotlib
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GCNConv, GATConv

sys.path.insert(0, '/home/sean429/swe3032')
from campus_graph import BUILDINGS, EDGES
from campus_synthetic import build_dataset, BUILDING_IDS

# ── 하이퍼파라미터 ────────────────────────────────────────────────────────────
N_INSTANCES = 200
HIDDEN      = 64
EPOCHS      = 200
LR          = 1e-3
TRAIN_RATIO = 0.8
SNAPSHOT_TRAIN = '/home/sean429/swe3032/data/2024_2_snapshot.csv'
SNAPSHOT_VAL   = '/home/sean429/swe3032/data/2024_2_snapshot.csv'
SNAPSHOT_TEST  = '/home/sean429/swe3032/data/2025_2_snapshot.csv'
OUT_CSV        = '/home/sean429/swe3032/results/2025_2_pred.csv'
OUT_PLOT       = '/home/sean429/swe3032/plots/campus_eval_{day}.png'
DAY_ORDER   = ['월', '화', '수', '목', '금', '토']

# ── 디바이스 ──────────────────────────────────────────────────────────────────
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'[device] {DEVICE}')

# ── 그래프 구조 (고정) ────────────────────────────────────────────────────────
node_idx   = {b: i for i, b in enumerate(BUILDING_IDS)}
max_dist   = max(dist for _, _, _, dist in EDGES)
edge_index = torch.tensor(
    [[node_idx[s], node_idx[d]] for s, d, _, _ in EDGES], dtype=torch.long
).t().contiguous().to(DEVICE)
edge_attr  = torch.tensor(
    [[w, dist / max_dist] for _, _, w, dist in EDGES], dtype=torch.float
).to(DEVICE)


def time_features(day: str, time: str) -> list:
    """요일 + 시각 → sin/cos 인코딩"""
    h, m   = map(int, time.split(':'))
    frac_h = (h * 60 + m) / (24 * 60)
    frac_d = DAY_ORDER.index(day) / len(DAY_ORDER) if day in DAY_ORDER else 0
    return [
        np.sin(2 * np.pi * frac_h), np.cos(2 * np.pi * frac_h),
        np.sin(2 * np.pi * frac_d), np.cos(2 * np.pi * frac_d),
    ]


def build_samples(instances: list, max_occ: float = None):
    """
    각 인스턴스 × 각 시간 전환(t→t+1) → PyG Data 리스트
    node feature: [occ_norm, sin_h, cos_h, sin_d, cos_d]  (5-dim)
    target:       occ_norm at t+1

    max_occ: 외부 지정 시 사용 — cross-semester 테스트 시 학습 기준으로 고정
    """
    all_occ = np.concatenate([inst[BUILDING_IDS].values for inst in instances])
    if max_occ is None:
        max_occ = float(all_occ.max()) if all_occ.max() > 0 else 1.0

    samples = []
    for inst in instances:
        rows = inst.reset_index(drop=True)
        for i in range(len(rows) - 1):
            r_t, r_t1 = rows.iloc[i], rows.iloc[i + 1]
            if r_t['요일'] != r_t1['요일']:
                continue
            tf = time_features(r_t['요일'], r_t['시각'])
            x  = torch.tensor([[r_t[b] / max_occ] + tf for b in BUILDING_IDS], dtype=torch.float)
            y  = torch.tensor([r_t1[b] / max_occ for b in BUILDING_IDS], dtype=torch.float)
            samples.append(Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y).to(DEVICE))

    return samples, max_occ


# ── 모델 정의 ─────────────────────────────────────────────────────────────────
IN_DIM = 5   # occ_norm + 4 time features

class MLP(nn.Module):
    """그래프 구조 무시 — 각 노드 독립 예측"""
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(IN_DIM, HIDDEN), nn.ReLU(),
            nn.Linear(HIDDEN, HIDDEN), nn.ReLU(),
            nn.Linear(HIDDEN, 1),
        )
    def forward(self, data):
        return self.net(data.x).squeeze(-1)


class GCN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = GCNConv(IN_DIM, HIDDEN)
        self.conv2 = GCNConv(HIDDEN, HIDDEN)
        self.head  = nn.Linear(HIDDEN, 1)
    def forward(self, data):
        x = F.relu(self.conv1(data.x, data.edge_index))
        x = F.relu(self.conv2(x,      data.edge_index))
        return self.head(x).squeeze(-1)


class GAT(nn.Module):
    def __init__(self, heads=4):
        super().__init__()
        self.conv1 = GATConv(IN_DIM,   HIDDEN // heads, heads=heads)
        self.conv2 = GATConv(HIDDEN,   HIDDEN,          heads=1)
        self.head  = nn.Linear(HIDDEN, 1)
    def forward(self, data):
        x = F.elu(self.conv1(data.x, data.edge_index))
        x = F.elu(self.conv2(x,      data.edge_index))
        return self.head(x).squeeze(-1)


# ── 학습 / 평가 ───────────────────────────────────────────────────────────────
BATCH_SIZE = 256

def train_model(model, train_data, val_data, epochs=EPOCHS):
    loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
    opt    = torch.optim.Adam(model.parameters(), lr=LR)
    for epoch in range(1, epochs + 1):
        model.train()
        total = 0.0
        for batch in loader:
            opt.zero_grad()
            loss = F.mse_loss(model(batch), batch.y)
            loss.backward()
            opt.step()
            total += loss.item()
        if epoch % 40 == 0:
            mae = evaluate(model, val_data)
            print(f'    epoch {epoch:3d} | train MSE {total/len(loader):.5f} | val MAE {mae:.4f}')


def evaluate(model, data_list):
    model.eval()
    loader = DataLoader(data_list, batch_size=BATCH_SIZE, shuffle=False)
    total, n = 0.0, 0
    with torch.no_grad():
        for batch in loader:
            total += F.l1_loss(model(batch), batch.y).item() * batch.num_graphs
            n     += batch.num_graphs
    return total / n


def predict_all_slots(models: dict, df: pd.DataFrame, max_occ: float) -> pd.DataFrame:
    """
    모든 모델로 2학기 snapshot 각 시간 전환(t→t+1) 예측.
    반환: (요일, 시각, building, actual, pred_MLP, pred_GCN, pred_GAT)
    """
    for m in models.values():
        m.eval()

    records = []
    with torch.no_grad():
        for day, day_df in df.groupby('요일', sort=False):
            day_df = day_df.reset_index(drop=True)
            for i in range(len(day_df) - 1):
                r_t, r_t1 = day_df.iloc[i], day_df.iloc[i + 1]
                tf   = time_features(str(r_t['요일']), str(r_t['시각']))
                x    = torch.tensor(
                    [[r_t[b] / max_occ] + tf for b in BUILDING_IDS], dtype=torch.float
                )
                data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr).to(DEVICE)
                preds = {name: model(data).cpu().numpy() for name, model in models.items()}

                for j, b in enumerate(BUILDING_IDS):
                    row = {'요일': day, '시각': r_t1['시각'], 'building': b,
                           'actual': float(r_t1[b])}
                    for name, p in preds.items():
                        row[f'pred_{name}'] = max(0.0, float(p[j]) * max_occ)
                    records.append(row)

    return pd.DataFrame(records)


# ── 시각화 ────────────────────────────────────────────────────────────────────
_BLDG_LABEL = {b: BUILDINGS[b]['name'] for b in BUILDING_IDS}
_DAY_EN     = {'월': 'Mon', '화': 'Tue', '수': 'Wed', '목': 'Thu', '금': 'Fri', '토': 'Sat'}


def plot_comparison(df_comp: pd.DataFrame, day: str, out_path: str):
    """선택 요일의 건물별 actual vs MLP/GCN/GAT 시계열 플롯"""
    sub = df_comp[df_comp['요일'] == day]
    if sub.empty:
        print(f'[WARN] {day}요일 데이터 없음 — 플롯 스킵')
        return

    MODEL_STYLE = {
        'MLP': dict(color='#3498db', ls='--', marker='s', lw=1.5, ms=3),
        'GCN': dict(color='#e67e22', ls='--', marker='^', lw=1.5, ms=3),
        'GAT': dict(color='#e74c3c', ls='-',  marker='o', lw=2.0, ms=4),
    }

    n_cols = 4
    n_rows = (len(BUILDING_IDS) + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows))
    axes = axes.flatten()

    for idx, b in enumerate(BUILDING_IDS):
        ax   = axes[idx]
        bsub = sub[sub['building'] == b].sort_values('시각')

        if bsub.empty or bsub['actual'].max() == 0:
            ax.text(0.5, 0.5, '수업 없음', ha='center', va='center',
                    transform=ax.transAxes, fontsize=11, color='#888')
            ax.set_title(f'{b}동  {_BLDG_LABEL[b]}', fontsize=10)
            ax.axis('off')
            continue

        times = bsub['시각'].values
        x_pos = np.arange(len(times))

        ax.fill_between(x_pos, bsub['actual'].values, alpha=0.12, color='black')
        ax.plot(x_pos, bsub['actual'].values, 'k-o', lw=2, ms=5, label='Actual', zorder=5)

        for name, style in MODEL_STYLE.items():
            col = f'pred_{name}'
            if col in bsub.columns:
                ax.plot(x_pos, bsub[col].values, label=name, zorder=4, **style)

        tick_step = max(1, len(times) // 6)
        ax.set_xticks(x_pos[::tick_step])
        ax.set_xticklabels(times[::tick_step], rotation=45, fontsize=7)
        ax.set_ylabel('명', fontsize=8)
        ax.set_title(f'{b}동  {_BLDG_LABEL[b]}\n(peak {bsub["actual"].max():.0f}명)', fontsize=9)
        ax.legend(fontsize=7, loc='upper right', framealpha=0.7)
        ax.grid(alpha=0.3)
        ax.set_xlim(-0.5, len(times) - 0.5)
        ax.set_ylim(bottom=0)

    for ax in axes[len(BUILDING_IDS):]:
        ax.axis('off')

    day_str = _DAY_EN.get(day, day)
    fig.suptitle(
        f'2025-2 {day_str} ({day}요일) — 건물별 혼잡도: 예측 vs 실제\n'
        f'학습: 2025-1학기 synthetic  |  테스트: 2025-2학기 실제',
        fontsize=13, fontweight='bold', y=1.01
    )
    plt.tight_layout()
    plt.savefig(out_path, dpi=120, bbox_inches='tight')
    plt.close()
    print(f'Saved: {out_path}')


# ── 메인 ─────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--eval-day', default='화', help='시각화할 요일 (default: 화)')
    args = parser.parse_args()

    # ── 1. 학습 데이터: 2024-1학기 synthetic augmentation (전부 train)
    print('▶ 2024-2 Synthetic 데이터 생성 중...')
    instances = build_dataset(SNAPSHOT_TRAIN, n_instances=N_INSTANCES)
    train_data, max_occ = build_samples(instances)
    np.random.shuffle(train_data)
    print(f'  train 샘플: {len(train_data)} | max_occ = {max_occ:.0f}명\n')

    # ── 2. val: 2024-1 실제 snapshot (synthetic 학습 결과 검증)
    df_val      = pd.read_csv(SNAPSHOT_VAL)
    val_data, _ = build_samples([df_val], max_occ=max_occ)
    print(f'  2024-1 실제 val 샘플: {len(val_data)}개')

    # ── 3. test: 2025-1 실제 snapshot (내년 동일 학기 예측 — cross-year)
    df_test         = pd.read_csv(SNAPSHOT_TEST)
    test_samples, _ = build_samples([df_test], max_occ=max_occ)
    print(f'  2025-1 실제 test 샘플: {len(test_samples)}개\n')

    # ── 3. 모델 학습 + 평가
    trained_models               = {}
    val_results, test_results    = {}, {}

    for name, model in [('MLP', MLP()), ('GCN', GCN()), ('GAT', GAT())]:
        model = model.to(DEVICE)
        print(f'▶ {name} 학습 중...')
        train_model(model, train_data, val_data)
        val_mae  = evaluate(model, val_data)    * max_occ
        test_mae = evaluate(model, test_samples) * max_occ
        val_results[name]     = val_mae
        test_results[name]    = test_mae
        trained_models[name]  = model
        print(f'  val  MAE (2024-2 실제) : {val_mae:.1f}명')
        print(f'  test MAE (2025-2 실제) : {test_mae:.1f}명\n')

    # ── 4. 결과 요약
    print('=' * 52)
    print(f'{"모델":<5} {"2024-2 val MAE":>17} {"2025-2 test MAE":>18}')
    print('-' * 52)
    for name in ('MLP', 'GCN', 'GAT'):
        print(f'{name:<5} {val_results[name]:>15.1f}명 {test_results[name]:>15.1f}명')
    print('=' * 52)

    best = min(test_results, key=test_results.get)
    print(f'\n2학기 최고 성능: {best}  (MAE {test_results[best]:.1f}명)')
    if test_results['GAT'] < test_results['MLP']:
        diff = test_results['MLP'] - test_results['GAT']
        print(f'GAT가 MLP보다 {diff:.1f}명 더 정확 → 그래프 구조가 cross-semester 일반화에 기여')
    else:
        print('(그래프 구조 효과 미미 — 노이즈 파라미터 또는 아키텍처 재검토 필요)')

    # ── 5. 슬롯별 예측값 CSV 저장
    print('\n▶ 2025-2 슬롯별 예측값 생성 중...')
    df_comp = predict_all_slots(trained_models, df_test, max_occ)
    df_comp.to_csv(OUT_CSV, index=False, encoding='utf-8-sig')
    print(f'  Saved: {OUT_CSV}  ({len(df_comp)}행)')

    # ── 6. 시각화
    out_plot = OUT_PLOT.format(day=args.eval_day)
    plot_comparison(df_comp, args.eval_day, out_plot)
