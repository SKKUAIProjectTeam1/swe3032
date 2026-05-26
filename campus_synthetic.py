"""
campus_synthetic.py
스케줄 기반 snapshot에 현실적 노이즈를 추가해 synthetic 데이터셋 생성.

노이즈 구성:
  1. 결석률  : Beta(8,2) ~ 평균 80%, 슬롯마다 독립
  2. 자습생  : Poisson(λ) — 수업 없어도 건물에 있는 학생
  3. 이동 지연: 직전 슬롯 인원의 일부가 아직 남아있음
  4. 측정 노이즈: Gaussian(0, σ)
"""
import numpy as np
import pandas as pd
from campus_graph import BUILDINGS

BUILDING_IDS = list(BUILDINGS.keys())


def generate_instance(df: pd.DataFrame,
                      seed: int = None,
                      attend_alpha: float = 8,
                      attend_beta:  float = 2,
                      bg_lambda:    float = 15,
                      lag_rate:     float = 0.08,
                      noise_std:    float = 5) -> pd.DataFrame:
    """
    snapshot DataFrame 한 장 → 노이즈 추가된 synthetic DataFrame 반환
    """
    rng = np.random.RandomState(seed)
    bld_cols = [c for c in df.columns if c in BUILDING_IDS]
    occ = df[bld_cols].values.astype(float)   # (T, B)
    T, B = occ.shape
    noisy = np.zeros((T, B))

    for t in range(T):
        for b in range(B):
            # 1. 결석률
            val = occ[t, b] * rng.beta(attend_alpha, attend_beta)
            # 2. 자습생
            val += rng.poisson(bg_lambda)
            # 3. 이동 지연
            if t > 0:
                val += noisy[t - 1, b] * lag_rate
            # 4. 측정 노이즈
            val += rng.normal(0, noise_std)
            noisy[t, b] = max(0.0, val)

    result = df.copy()
    result[bld_cols] = np.round(noisy).astype(int)
    return result


def build_dataset(snapshot_path: str,
                  n_instances: int = 200,
                  seed: int = 42) -> list[pd.DataFrame]:
    """
    N개의 synthetic 인스턴스 반환 (각각 다른 seed)
    """
    df = pd.read_csv(snapshot_path)
    rng = np.random.RandomState(seed)
    seeds = rng.randint(0, 100_000, size=n_instances)
    return [generate_instance(df, s) for s in seeds]


if __name__ == '__main__':
    instances = build_dataset('/home/sean429/swe3032/2025_1_snapshot.csv', n_instances=5)
    orig = pd.read_csv('/home/sean429/swe3032/2025_1_snapshot.csv')

    print('=== 원본 vs synthetic 비교 (월 10:00) ===')
    t = orig[(orig['요일'] == '월') & (orig['시각'] == '10:00')].iloc[0]
    print('원본  :', t[BUILDING_IDS].to_dict())
    for i, inst in enumerate(instances):
        row = inst[(inst['요일'] == '월') & (inst['시각'] == '10:00')].iloc[0]
        print(f'inst{i} :', row[BUILDING_IDS].to_dict())
