"""
campus_graph.py
캠퍼스 건물 그래프 — 건물 노드 및 이동 엣지

실제 캠퍼스 구조 (북→남):
  85동(최북단)  →  26동(25/26/27 클러스터 중심)  →  22동(허브)  →  주차장3  →  33/40동
  22동 지선: 23동(북서), 21동(서)
  ※ 25/27동은 데이터 없음 — 26동 클러스터로 통합 표현
  ※ 주차장3은 22↔33 경로의 경유지 — 별도 노드 없이 엣지 거리로 반영

좌표계: 카카오맵 픽셀 기반, y축 상향(북=+)
"""

# ── 건물 노드 ──────────────────────────────────────────────────────────────────
BUILDINGS = {
    '85': {'name': '산학협력센터',  'campus_x':   0, 'campus_y':    0},   # 최북단, ㄷ자형
    '26': {'name': '제2공학관',     'campus_x': 220, 'campus_y': -100},   # 25/26/27 클러스터 중심
    '22': {'name': '제1공학관',     'campus_x': 210, 'campus_y': -220},   # 중앙 허브
    '23': {'name': '공과대학',      'campus_x': 110, 'campus_y': -165},   # 22동 북서쪽
    '21': {'name': '정보통신대학',  'campus_x': 105, 'campus_y': -245},   # 22동 서쪽
    '33': {'name': '화학관',        'campus_x': 185, 'campus_y': -430},   # 주차장3 남쪽
    '40': {'name': '반도체관',      'campus_x': 250, 'campus_y': -455},   # 33동 연결 복합동
}

# ── 건물 간 이동 엣지 ──────────────────────────────────────────────────────────
# (from, to, weight, walk_distance_m)
# 모든 연결은 양방향 — 가중치 동일 (비대칭 이동 패턴은 GAT attention이 학습)
EDGES = [
    # ── 메인 남북 축: 85 ↔ 26 ↔ 22 ──────────────────────────────────────────
    ('85', '26', 0.6, 220),
    ('26', '85', 0.6, 220),
    ('26', '22', 0.6, 150),   # 26동(클러스터 정문) → 22동(허브) 직통
    ('22', '26', 0.6, 150),

    # ── 22동 지선: 23동(북서), 21동(서) ───────────────────────────────────────
    ('22', '23', 0.5, 130),
    ('23', '22', 0.5, 130),
    ('22', '21', 0.5,  80),
    ('21', '22', 0.5,  80),

    # ── 남단 복합동: 22 → 주차장3 → 33↔40 ────────────────────────────────────
    ('22', '33', 0.4, 210),   # 주차장3 경유
    ('33', '22', 0.4, 210),
    ('33', '40', 0.5,  40),   # 33/40 문 공유 — 단일 진입점
    ('40', '33', 0.5,  40),

    # ── 대각선 / 우회 경로 (수업 배치에 따라 발생) ────────────────────────────
    ('85', '22', 0.2, 340),   # 85동 → 22동 직통 (ㄷ자 날개 통해 대각선)
    ('22', '85', 0.2, 340),
    ('26', '23', 0.3, 160),   # 26동에서 23동으로 직접 접근 가능
    ('23', '26', 0.3, 160),
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
    """스냅샷 CSV 한 행 → PyTorch Geometric Data"""
    try:
        import torch
        from torch_geometric.data import Data
    except ImportError:
        raise ImportError("pip install torch torch-geometric")

    node_ids = list(BUILDINGS.keys())
    idx_map  = {b: i for i, b in enumerate(node_ids)}
    x = torch.tensor(
        [[float(snapshot_row.get(b, 0))] for b in node_ids], dtype=torch.float
    )
    valid  = [(s, d, w, dist) for s, d, w, dist in EDGES if s in idx_map and d in idx_map]
    ei     = torch.tensor([[idx_map[s], idx_map[d]] for s, d, _, _ in valid],
                          dtype=torch.long).t().contiguous()
    max_d  = max(dist for _, _, _, dist in valid)
    ea     = torch.tensor([[w, dist / max_d] for _, _, w, dist in valid], dtype=torch.float)
    return Data(x=x, edge_index=ei, edge_attr=ea)


if __name__ == '__main__':
    G = build_nx_graph()
    print(f'노드 {G.number_of_nodes()}개: {list(G.nodes())}')
    print(f'\n엣지 {G.number_of_edges()}개:')
    for u, v, d in G.edges(data=True):
        print(f'  {u} → {v}  w={d["entrance_weight"]}  dist={d["distance"]}m')
