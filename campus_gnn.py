"""
campus_gnn.py
건물 간 이동 그래프 위에서 다음 슬롯 체류 인원 예측.

비교:
  MLP  — 그래프 구조 무시, 각 건물 독립적으로 예측
  GCN  — 이웃 건물 정보를 집계해 예측
  GAT  — 이웃마다 attention weight 학습

실행: python campus_gnn.py
"""
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv, GATConv

sys.path.insert(0, '/home/sean429/swe3032')
from campus_graph import BUILDINGS, EDGES
from campus_synthetic import build_dataset, BUILDING_IDS

# ── 하이퍼파라미터 ────────────────────────────────────────────────────────────
N_INSTANCES  = 200
HIDDEN       = 64
EPOCHS       = 200
LR           = 1e-3
TRAIN_RATIO  = 0.8
SNAPSHOT_PATH = '/home/sean429/swe3032/2025_1_snapshot.csv'
DAY_ORDER    = ['월', '화', '수', '목', '금', '토']

# ── 그래프 구조 (고정) ────────────────────────────────────────────────────────
node_idx = {b: i for i, b in enumerate(BUILDING_IDS)}
max_dist = max(dist for _, _, _, dist in EDGES)

edge_index = torch.tensor(
    [[node_idx[s], node_idx[d]] for s, d, _, _ in EDGES], dtype=torch.long
).t().contiguous()

edge_attr = torch.tensor(
    [[w, dist / max_dist] for _, _, w, dist in EDGES], dtype=torch.float
)


def time_features(day: str, time: str) -> list:
    """요일 + 시각 → sin/cos 인코딩"""
    h, m   = map(int, time.split(':'))
    frac_h = (h * 60 + m) / (24 * 60)
    frac_d = DAY_ORDER.index(day) / len(DAY_ORDER) if day in DAY_ORDER else 0
    return [
        np.sin(2 * np.pi * frac_h),
        np.cos(2 * np.pi * frac_h),
        np.sin(2 * np.pi * frac_d),
        np.cos(2 * np.pi * frac_d),
    ]


def build_samples(instances: list) -> list[Data]:
    """
    각 인스턴스 × 각 시간 전환(t→t+1) → PyG Data 리스트
    node feature: [occ_norm, sin_h, cos_h, sin_d, cos_d]  (5-dim)
    target:       occ_norm at t+1
    """
    samples = []
    # 전체 max occupancy (정규화 기준)
    all_occ = np.concatenate([
        inst[BUILDING_IDS].values for inst in instances
    ])
    max_occ = all_occ.max() if all_occ.max() > 0 else 1.0

    for inst in instances:
        rows = inst.reset_index(drop=True)
        for i in range(len(rows) - 1):
            r_t   = rows.iloc[i]
            r_t1  = rows.iloc[i + 1]

            # 같은 요일 내 전환만 사용
            if r_t['요일'] != r_t1['요일']:
                continue

            tf = time_features(r_t['요일'], r_t['시각'])
            x  = torch.tensor(
                [[r_t[b] / max_occ] + tf for b in BUILDING_IDS],
                dtype=torch.float
            )
            y  = torch.tensor(
                [r_t1[b] / max_occ for b in BUILDING_IDS],
                dtype=torch.float
            )
            samples.append(Data(
                x=x, edge_index=edge_index, edge_attr=edge_attr, y=y
            ))

    return samples, max_occ


# ── 모델 정의 ─────────────────────────────────────────────────────────────────
IN_DIM = 5   # occ + 4 time features

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
        self.conv1 = GCNConv(IN_DIM,  HIDDEN)
        self.conv2 = GCNConv(HIDDEN,  HIDDEN)
        self.head  = nn.Linear(HIDDEN, 1)

    def forward(self, data):
        x = F.relu(self.conv1(data.x, data.edge_index))
        x = F.relu(self.conv2(x,      data.edge_index))
        return self.head(x).squeeze(-1)


class GAT(nn.Module):
    def __init__(self, heads=4):
        super().__init__()
        self.conv1 = GATConv(IN_DIM,        HIDDEN // heads, heads=heads)
        self.conv2 = GATConv(HIDDEN,        HIDDEN,          heads=1)
        self.head  = nn.Linear(HIDDEN, 1)

    def forward(self, data):
        x = F.elu(self.conv1(data.x, data.edge_index))
        x = F.elu(self.conv2(x,      data.edge_index))
        return self.head(x).squeeze(-1)


# ── 학습 / 평가 ───────────────────────────────────────────────────────────────
def train_model(model, train_data, test_data, epochs=EPOCHS):
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    train_losses, test_maes = [], []

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0
        for data in train_data:
            opt.zero_grad()
            pred = model(data)
            loss = F.mse_loss(pred, data.y)
            loss.backward()
            opt.step()
            total_loss += loss.item()
        train_losses.append(total_loss / len(train_data))

        if epoch % 20 == 0:
            mae = evaluate(model, test_data)
            test_maes.append((epoch, mae))
            print(f'  epoch {epoch:3d} | train MSE {train_losses[-1]:.5f} | test MAE {mae:.4f}')

    return train_losses, test_maes


def evaluate(model, data_list):
    model.eval()
    total_mae = 0
    with torch.no_grad():
        for data in data_list:
            pred = model(data)
            total_mae += F.l1_loss(pred, data.y).item()
    return total_mae / len(data_list)


# ── 메인 ─────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    print('▶ Synthetic 데이터 생성 중...')
    instances = build_dataset(SNAPSHOT_PATH, n_instances=N_INSTANCES)
    samples, max_occ = build_samples(instances)

    np.random.shuffle(samples)
    split      = int(len(samples) * TRAIN_RATIO)
    train_data = samples[:split]
    test_data  = samples[split:]
    print(f'  총 {len(samples)}개 샘플 | train {len(train_data)} / test {len(test_data)}')
    print(f'  max_occ = {max_occ:.0f}명\n')

    results = {}
    for name, model in [('MLP', MLP()), ('GCN', GCN()), ('GAT', GAT())]:
        print(f'▶ {name} 학습 중...')
        _, test_maes = train_model(model, train_data, test_data)
        final_mae = evaluate(model, test_data)
        results[name] = final_mae * max_occ   # 실제 인원 단위로 환산
        print(f'  → 최종 test MAE: {results[name]:.1f}명\n')

    print('=' * 40)
    print('모델별 최종 test MAE (명)')
    for name, mae in results.items():
        bar = '█' * int(mae / 5)
        print(f'  {name:4s}: {mae:6.1f}명  {bar}')
    print('=' * 40)

    best = min(results, key=results.get)
    print(f'\n최고 성능: {best}  (MAE {results[best]:.1f}명)')
    if results['GCN'] < results['MLP']:
        diff = results['MLP'] - results['GCN']
        print(f'GCN이 MLP보다 {diff:.1f}명 더 정확 → 그래프 구조가 도움됨')
    else:
        print('(그래프 구조 효과 미미 — 노이즈 파라미터 조정 필요)')
