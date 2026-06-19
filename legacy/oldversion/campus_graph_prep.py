"""
campus_graph_prep.py
14개 노드(병합) 기준으로
  1. 건물쌍별 "전환압력" 점수 (시간표상 A종료->B시작이 맞물리는 누적 잠재인원)
  2. 건물쌍별 물리적 거리 (campus_topdown 좌표의 픽셀 거리, 병합그룹은 centroid)
를 계산해서 거리-전환압력 분포를 살펴보기 위한 초안.

거리 컷오프 기준은 아직 정하지 않음 — 이 분포를 보고 결정.
"""
import os, sys
from collections import defaultdict
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

matplotlib.rcParams['font.family'] = 'Malgun Gothic'
matplotlib.rcParams['axes.unicode_minus'] = False

sys.path.insert(0, os.path.dirname(__file__))
from campus_snapshot import load_semester, parse_sessions, SEMESTER_FOLDERS, _t2m

TRANSITION_WINDOW = 15  # 분 — 표준 쉬는시간

EXCLUDE = {'05'}
BUILDING_MAP = {
    '25': '25_26_27', '26': '25_26_27', '27': '25_26_27',
    '51': '51_61_62', '61': '51_61_62', '62': '51_61_62',
    '31': '31_32',    '32': '31_32',
    '62B08': '83',
}

# campus_topdown 좌표 (회전 보정 전 원본 픽셀, make_topdown_map.py BUILDINGS_RAW 동일)
COORDS_RAW = {
    '27': (617, 175), '26': (649, 230), '25': (618, 263),
    '23': (582, 287), '22': (655, 362), '21': (600, 407),
    '24': (542, 448), '85': (509, 114), '61': (352,  97),
    '62': (392,  97), '51': (347, 160), '32': (430, 227),
    '31': (444, 288), '71': (263, 551), '86': (569, 591),
    '53': (638, 614), '81': (752, 574), '83': (762, 447),
    '33': (743, 641), '40': (820, 641),
}


def remap(b):
    if b in EXCLUDE:
        return None
    return BUILDING_MAP.get(b, b)


def collect_events():
    """4개 학기 전체 세션 이벤트 수집 (병합매핑 적용 후)"""
    events = []
    for key, folder in SEMESTER_FOLDERS.items():
        df = load_semester(folder)
        if df.empty:
            continue
        for _, row in df.iterrows():
            for s in parse_sessions(row.get('수업시간대', ''), str(row.get('building', ''))):
                b = remap(s['building'])
                if b is None:
                    continue
                events.append({
                    'day': s['day'],
                    'start': _t2m(s['start']),
                    'end': _t2m(s['end']),
                    'building': b,
                    'enr': int(row.get('분반인원', 0)),
                })
    return events


def transition_scores(events, window=TRANSITION_WINDOW):
    """
    (A,B) -> 누적 점수.
    A에서 끝나는 세션의 종료시각과 B에서 시작하는 세션의 시작시각 차이가
    [0, window]분 이내이면 min(A인원, B인원)을 누적.
    """
    scores = defaultdict(float)
    for day in set(e['day'] for e in events):
        day_ev = [e for e in events if e['day'] == day]
        enders   = [e for e in day_ev]
        starters = [e for e in day_ev]
        for ea in enders:
            for eb in starters:
                if ea['building'] == eb['building']:
                    continue
                gap = eb['start'] - ea['end']
                if 0 <= gap <= window:
                    scores[(ea['building'], eb['building'])] += min(ea['enr'], eb['enr'])
    return scores


def building_activity(events):
    """건물별 '총 점유 인원-시간' (enr × duration) — 정규화 기준."""
    act = defaultdict(float)
    for e in events:
        act[e['building']] += e['enr'] * (e['end'] - e['start'])
    return act


def node_coords(active_nodes):
    """병합그룹은 멤버 건물 좌표의 centroid. 실제 강의데이터에 있는 노드만 포함."""
    groups = defaultdict(list)
    for code, xy in COORDS_RAW.items():
        node = remap(code)
        if node is not None and node in active_nodes:
            groups[node].append(xy)
    return {node: tuple(np.mean(pts, axis=0)) for node, pts in groups.items()}


def main():
    print('이벤트 수집 중...')
    events = collect_events()
    print(f'  총 {len(events)}개 세션 이벤트 (4개학기 누적, 병합/제외 적용)')
    active_nodes = set(e['building'] for e in events)
    print(f'  실제 강의 있는 노드 {len(active_nodes)}개: {sorted(active_nodes)}')

    print('전환압력 점수 계산 중...')
    scores = transition_scores(events)
    activity = building_activity(events)

    coords = node_coords(active_nodes)
    nodes = sorted(coords)
    print(f'  노드 {len(nodes)}개: {nodes}')

    rows = []
    for i, a in enumerate(nodes):
        for b in nodes[i+1:]:
            d = float(np.hypot(coords[a][0]-coords[b][0], coords[a][1]-coords[b][1]))
            pressure = scores.get((a, b), 0) + scores.get((b, a), 0)
            denom = activity[a] * activity[b]
            ratio = pressure / denom if denom else 0.0
            rows.append({'A': a, 'B': b, '거리_px': round(d, 1),
                         '전환압력_누적': round(pressure, 1),
                         '정규화_비율': ratio})

    edge_df = pd.DataFrame(rows).sort_values('거리_px').reset_index(drop=True)
    # 비교하기 쉽게 정규화 비율을 0~100 스케일로 변환
    mx = edge_df['정규화_비율'].max()
    edge_df['정규화_점수'] = (edge_df['정규화_비율'] / mx * 100).round(2) if mx else 0.0
    out_csv = os.path.join(os.path.dirname(__file__), 'graph_edge_draft.csv')
    edge_df.to_csv(out_csv, index=False, encoding='utf-8-sig')
    print(f'저장: {out_csv}')

    # 거리 vs (원시 / 정규화) 점수 산점도 - 둘을 나란히 비교
    fig, axes = plt.subplots(1, 2, figsize=(18, 7))

    ax = axes[0]
    ax.scatter(edge_df['거리_px'], edge_df['전환압력_누적'], s=40, alpha=0.7, color='#4a90d9', edgecolor='#1a5fa8')
    for _, r in edge_df.iterrows():
        ax.annotate(f"{r['A']}-{r['B']}", (r['거리_px'], r['전환압력_누적']),
                    fontsize=6, xytext=(3, 3), textcoords='offset points')
    ax.set_xlabel('건물쌍 거리 (px)')
    ax.set_ylabel('원시 누적 전환압력 (min(A인원,B인원) 합)')
    ax.set_title('(1) 거리 vs 원시 전환압력\n(건물 활동량에 좌우될 수 있음)')
    ax.grid(alpha=0.3)

    ax2 = axes[1]
    ax2.scatter(edge_df['거리_px'], edge_df['정규화_점수'], s=40, alpha=0.7, color='#27ae60', edgecolor='#1a7a45')
    for _, r in edge_df.iterrows():
        ax2.annotate(f"{r['A']}-{r['B']}", (r['거리_px'], r['정규화_점수']),
                     fontsize=6, xytext=(3, 3), textcoords='offset points')
    ax2.set_xlabel('건물쌍 거리 (px)')
    ax2.set_ylabel('활동량 정규화 점수 (0~100, 전환압력/(활동A x 활동B))')
    ax2.set_title('(2) 거리 vs 정규화 점수\n(건물별 전체 활동량으로 나눠 "기대 대비 초과 빈도"만 봄)')
    ax2.grid(alpha=0.3)

    plt.tight_layout()
    out_png = os.path.join(os.path.dirname(__file__), 'graph_edge_draft.png')
    plt.savefig(out_png, dpi=150, bbox_inches='tight')
    print(f'저장: {out_png}')

    corr_raw  = edge_df['거리_px'].corr(edge_df['전환압력_누적'])
    corr_norm = edge_df['거리_px'].corr(edge_df['정규화_점수'])

    # 요약 (콘솔 cp949 깨짐 방지를 위해 UTF-8 파일로도 저장)
    nz = edge_df[edge_df['전환압력_누적'] > 0]
    summary = []
    summary.append(f'노드 {len(nodes)}개: {nodes}')
    summary.append(f'전체 쌍 {len(edge_df)}개 중 전환압력 > 0 인 쌍: {len(nz)}개')
    summary.append(f'거리-원시점수 상관계수: {corr_raw:.3f}')
    summary.append(f'거리-정규화점수 상관계수: {corr_norm:.3f}')
    summary.append('')
    summary.append('[건물별 활동량 (enr x duration 누적)]')
    for b in nodes:
        summary.append(f'  {b}: {activity[b]:.0f}')
    summary.append('')
    summary.append('[거리 가까운 순 15개 쌍 - 원시 vs 정규화 비교]')
    summary.append(edge_df.sort_values('거리_px', ascending=True).head(15)[['A','B','거리_px','전환압력_누적','정규화_점수']].to_string(index=False))
    summary.append('')
    summary.append('[거리 먼 순 15개 쌍 - 원시 vs 정규화 비교]')
    summary.append(edge_df.sort_values('거리_px', ascending=False).head(15)[['A','B','거리_px','전환압력_누적','정규화_점수']].to_string(index=False))
    summary.append('')
    summary.append('[정규화 점수 높은 순 15개 쌍]')
    summary.append(edge_df.sort_values('정규화_점수', ascending=False).head(15)[['A','B','거리_px','전환압력_누적','정규화_점수']].to_string(index=False))
    summary.append('')
    summary.append('[정규화 점수 낮은 순 15개 쌍]')
    summary.append(edge_df.sort_values('정규화_점수', ascending=True).head(15)[['A','B','거리_px','전환압력_누적','정규화_점수']].to_string(index=False))
    summary.append('')
    summary.append('[전체 91개 쌍 - 거리순 정렬]')
    summary.append(edge_df.to_string(index=False))

    import io
    summary_path = os.path.join(os.path.dirname(__file__), 'graph_edge_draft_summary.txt')
    with io.open(summary_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(summary))
    print(f'요약 저장: {summary_path}')


if __name__ == '__main__':
    main()
