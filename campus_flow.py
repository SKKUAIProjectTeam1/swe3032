"""
campus_flow.py
시간 슬롯 전환(t → t+1)마다 건물 간 이동 인원(엣지 플로우) 계산.

flow(u→v, t) = departure[u][t] × edge_weight(u→v)
  departure[u][t] = max(0, occ[u][t] - occ[u][t+1])
"""
import pandas as pd
from campus_graph import EDGES, BUILDINGS

BUILDING_IDS = list(BUILDINGS.keys())

# (src, dst) → entrance_weight
EDGE_WEIGHT = {(s, d): w for s, d, w, _ in EDGES}
EDGE_DIST   = {(s, d): dist for s, d, _, dist in EDGES}


def compute_flows(df: pd.DataFrame) -> pd.DataFrame:
    """
    snapshot CSV(요일×시각×건물) → 엣지 플로우 DataFrame
    columns: 요일, 시각, from, to, flow, distance
    """
    records = []

    for day, day_df in df.groupby('요일'):
        day_df = day_df.reset_index(drop=True)

        for i in range(len(day_df) - 1):
            row_t  = day_df.iloc[i]
            row_t1 = day_df.iloc[i + 1]
            time   = row_t['시각']

            for bld in BUILDING_IDS:
                occ_t  = int(row_t.get(bld,  0))
                occ_t1 = int(row_t1.get(bld, 0))
                dep    = max(0, occ_t - occ_t1)
                if dep == 0:
                    continue

                for (src, dst), w in EDGE_WEIGHT.items():
                    if src != bld:
                        continue
                    records.append({
                        '요일': day, '시각': time,
                        'from': src, 'to': dst,
                        'flow': dep * w,
                        'distance': EDGE_DIST[(src, dst)],
                    })

    return pd.DataFrame(records)


def flow_lookup(flows_df: pd.DataFrame) -> dict:
    """
    (요일, 시각, from, to) → flow  빠른 조회용 dict
    """
    return {
        (r['요일'], r['시각'], r['from'], r['to']): r['flow']
        for _, r in flows_df.iterrows()
    }


if __name__ == '__main__':
    df    = pd.read_csv('/home/sean429/swe3032/2025_1_snapshot.csv')
    flows = compute_flows(df)
    print(flows[flows['flow'] > 0].sort_values('flow', ascending=False).head(20).to_string(index=False))
    print(f'\n총 {len(flows)}개 레코드, 최대 flow = {flows["flow"].max():.0f}명')
