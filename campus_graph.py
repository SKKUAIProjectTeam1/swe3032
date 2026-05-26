"""
campus_graph.py
캠퍼스 건물 그래프 정의 — 건물 노드, 출입구 가중치, 건물 간 엣지

좌표계: 카카오맵 픽셀 × 1.5m/px, 85동 원점(0,0), y축 상향(북=+)
캠퍼스 주축(남북): 85동(북) → 26동 → 23동 → 22동 → 21동 → [주차장] → 33동 → 40동(남)
"""

# ── 건물 노드 ──────────────────────────────────────────────────────────────────
BUILDINGS = {
    '21': {'name': '정보통신대학',  'campus_x': 210, 'campus_y': -300},  # 22동 남쪽, 클러스터 끝
    '22': {'name': '제1공학관',     'campus_x': 195, 'campus_y': -255},  # 클러스터 허브
    '23': {'name': '공과대학',      'campus_x': 197, 'campus_y': -175},  # 클러스터 북단, 주 진입구
    '26': {'name': '제2공학관',     'campus_x': 195, 'campus_y':  -95},  # 25/26/27 클러스터 정문
    '33': {'name': '화학관',        'campus_x': 185, 'campus_y': -480},  # 주차장 남쪽
    '40': {'name': '반도체관',      'campus_x': 235, 'campus_y': -510},  # 33동과 연결된 복합동
    '85': {'name': '산학협력센터',  'campus_x':   0, 'campus_y':    0},  # 캠퍼스 북부, ㄷ자형
}

# ── 출입구 정의 ────────────────────────────────────────────────────────────────
# weight 합 = 1.0 (건물에서 나가는 인원의 출입구별 분배 비율)
# faces: 이 문으로 나갔을 때 도달하는 인접 건물
ENTRANCES = {
    # ── 21동 (정보통신대학) ────────────────────────────────────────────────────
    # 22동 북쪽에 연결. 외부 독립 출입구 있음
    '21': [
        {'id': '21_main', 'label': '정문', 'weight': 0.6, 'direction': '북', 'faces': ['22']},
        {'id': '21_back', 'label': '후문', 'weight': 0.2, 'direction': '남', 'faces': []},
        {'id': '21_side', 'label': '측문', 'weight': 0.2, 'direction': '서', 'faces': []},
    ],

    # ── 22동 (제1공학관) ───────────────────────────────────────────────────────
    # 21·23동과 내부 연결. 정문=북(23동·26동 방향), 후문=남
    '22': [
        {'id': '22_main', 'label': '정문', 'weight': 0.6, 'direction': '북', 'faces': ['23']},
        {'id': '22_back', 'label': '후문', 'weight': 0.2, 'direction': '남', 'faces': ['33']},
        {'id': '22_side', 'label': '측문', 'weight': 0.2, 'direction': '동', 'faces': ['21']},
    ],

    # ── 23동 (공과대학) ────────────────────────────────────────────────────────
    # 클러스터 북단 = 주 진입구. "23동으로 많이 들어가"
    # 정문=북(26동 방향 메인 도로), 후문=남(22동 내부 연결)
    '23': [
        {'id': '23_main', 'label': '정문', 'weight': 0.6, 'direction': '북', 'faces': ['26']},
        {'id': '23_back', 'label': '후문', 'weight': 0.2, 'direction': '남', 'faces': ['22']},
        {'id': '23_side', 'label': '측문', 'weight': 0.2, 'direction': '동', 'faces': []},
    ],

    # ── 26동 (제2공학관, 25/26/27 클러스터 정문) ──────────────────────────────
    # 정문=북서(85동 방향), 후문=남(23동 방향)
    # 클러스터 내부: 27동↔26동↔25동 순서
    '26': [
        {'id': '26_main', 'label': '정문', 'weight': 0.6, 'direction': '북서', 'faces': ['85']},
        {'id': '26_back', 'label': '후문', 'weight': 0.2, 'direction': '남',   'faces': ['23']},
        {'id': '26_side', 'label': '측문', 'weight': 0.2, 'direction': '동',   'faces': []},
    ],

    # ── 85동 (산학협력센터, ㄷ자형) ───────────────────────────────────────────
    # 아래가 열린 ㄷ자: 남쪽(26동·23동 방향)이 정문, 동서 양 날개에 측문
    '85': [
        {'id': '85_main', 'label': '정문', 'weight': 0.6, 'direction': '남',  'faces': ['26']},
        {'id': '85_back', 'label': '후문', 'weight': 0.2, 'direction': '북',  'faces': []},
        {'id': '85_side', 'label': '측문', 'weight': 0.2, 'direction': '동',  'faces': ['23']},
    ],

    # ── 33동 (약학관) ─────────────────────────────────────────────────────────
    # 40동과 연결된 복합동. 정문=북(주차장 건너 22동 방향)
    '33': [
        {'id': '33_main', 'label': '정문', 'weight': 0.6, 'direction': '북', 'faces': ['22']},
        {'id': '33_back', 'label': '후문', 'weight': 0.2, 'direction': '남', 'faces': []},
        {'id': '33_side', 'label': '측문', 'weight': 0.2, 'direction': '동', 'faces': ['40']},  # 40동 내부 연결
    ],

    # ── 40동 (한독관) ─────────────────────────────────────────────────────────
    # 33동과 연결. 정문=북서(33동 통해 22동 방향)
    '40': [
        {'id': '40_main', 'label': '정문', 'weight': 0.6, 'direction': '서', 'faces': ['33']},
        {'id': '40_back', 'label': '후문', 'weight': 0.2, 'direction': '동', 'faces': []},
        {'id': '40_side', 'label': '측문', 'weight': 0.2, 'direction': '북', 'faces': []},
    ],
}

# ── 건물 간 방향 그래프 엣지 ──────────────────────────────────────────────────
# (from, to, entrance_weight, walk_distance_m)
EDGES = [
    # ── 21/22/23 클러스터 내부 (연결 건물군) ─────────────────────────────────
    ('21', '22',  0.6,  25),   # 21동 정문(북) → 22동
    ('22', '21',  0.2,  25),   # 22동 측문(동) → 21동
    ('22', '23',  0.6,  30),   # 22동 정문(북) → 23동
    ('23', '22',  0.2,  30),   # 23동 후문(남) → 22동

    # ── 23동(클러스터 북단) ↔ 26동 ───────────────────────────────────────────
    # "23동으로 많이 들어가" → 26동에서 오는 사람이 23동으로 주로 진입
    ('23', '26',  0.6, 105),   # 23동 정문(북) → 26동
    ('26', '23',  0.2, 105),   # 26동 후문(남) → 23동

    # ── 26동 ↔ 85동 (메인 이동축) ────────────────────────────────────────────
    ('26', '85',  0.6, 220),
    ('85', '26',  0.6, 220),

    # ── 85동 측문 → 23동 직통 (ㄷ자 동쪽 날개) ───────────────────────────────
    ('85', '23',  0.2, 290),

    # ── 22동 후문 ↔ 33동 (주차장 통과, 거리 있음) ────────────────────────────
    ('22', '33',  0.2, 230),
    ('33', '22',  0.6, 230),

    # ── 33동 ↔ 40동 (연결 복합동, 내부 통로) ─────────────────────────────────
    ('33', '40',  0.2,  40),
    ('40', '33',  0.6,  40),
]


# ── 유틸 ──────────────────────────────────────────────────────────────────────

def build_nx_graph():
    """EDGES → networkx DiGraph (시각화·검증용)"""
    import networkx as nx
    G = nx.DiGraph()
    for bld_id, attrs in BUILDINGS.items():
        G.add_node(bld_id, **attrs)
    for src, dst, w, dist in EDGES:
        G.add_edge(src, dst, entrance_weight=w, distance=dist)
    return G


def build_pyg_data(snapshot_row: dict):
    """
    스냅샷 CSV 한 행 (건물별 체류 인원) → PyTorch Geometric Data

    노드 피처: [occupancy(t)]  ← 추후: Δ(t), sin/cos(시각), 요일 원핫
    엣지 피처: [entrance_weight, distance_norm]
    """
    try:
        import torch
        from torch_geometric.data import Data
    except ImportError:
        raise ImportError("pip install torch torch-geometric")

    node_ids = list(BUILDINGS.keys())
    idx_map  = {b: i for i, b in enumerate(node_ids)}

    x = torch.tensor(
        [[float(snapshot_row.get(b, 0))] for b in node_ids],
        dtype=torch.float
    )

    valid_edges = [(s, d, w, dist) for s, d, w, dist in EDGES
                   if s in idx_map and d in idx_map]

    edge_index = torch.tensor(
        [[idx_map[s], idx_map[d]] for s, d, _, _ in valid_edges],
        dtype=torch.long
    ).t().contiguous()

    max_dist = max(dist for _, _, _, dist in valid_edges)
    edge_attr = torch.tensor(
        [[w, dist / max_dist] for _, _, w, dist in valid_edges],
        dtype=torch.float
    )

    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr)


if __name__ == '__main__':
    G = build_nx_graph()
    print(f'노드 {G.number_of_nodes()}개: {list(G.nodes())}')
    print(f'\n엣지 {G.number_of_edges()}개:')
    for u, v, d in G.edges(data=True):
        print(f'  {u:4s} → {v:4s}  weight={d["entrance_weight"]}  dist={d["distance"]}m')

    print('\n[엣지 요약 — 방향별]')
    print('  남북 주축: 85 ↔ 26 ↔ 23 ↔ 22 ↔ 21')
    print('  남단 복합: 22 → 33 ↔ 40  (주차장 구간)')
    print('  클러스터: 21-22-23 내부 연결')
