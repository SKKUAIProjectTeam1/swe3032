"""
campus_graph.py
캠퍼스 건물 그래프 및 도로망 (Junction-aware)
"""
import numpy as np
import networkx as nx

# ── 건물 노드 (Bounding Box Center) ──────────────────────────────────────────
BUILDINGS = {
    '03': {'name': '학생회관', 'campus_x': 762, 'campus_y': 1040},
    '05': {'name': '수성관', 'campus_x': 419, 'campus_y': 1281},
    '21': {'name': '정보통신대학', 'campus_x': 1454, 'campus_y': 1173},
    '22': {'name': '제1공학관', 'campus_x': 1603, 'campus_y': 1117},
    '23': {'name': '공과대학', 'campus_x': 1496, 'campus_y': 962},
    '24': {'name': '공학실습동', 'campus_x': 1808, 'campus_y': 1250},
    '25': {'name': '제2공학관(25)', 'campus_x': 1545, 'campus_y': 838},
    '26': {'name': '제2공학관(26)', 'campus_x': 1689, 'campus_y': 741},
    '27': {'name': '제2공학관(27)', 'campus_x': 1565, 'campus_y': 606},
    '31': {'name': '제1과학관', 'campus_x': 1139, 'campus_y': 915},
    '32': {'name': '제2과학관', 'campus_x': 1056, 'campus_y': 777},
    '33': {'name': '화학관', 'campus_x': 1612, 'campus_y': 1798},
    '40': {'name': '반도체관', 'campus_x': 1800, 'campus_y': 1748},
    '48': {'name': '삼성학술정보관', 'campus_x': 1083, 'campus_y': 1081},
    '51': {'name': '기초학문관', 'campus_x': 945, 'campus_y': 630},
    '53': {'name': '약학관', 'campus_x': 1501, 'campus_y': 1679},
    '61': {'name': '생명공학관(61)', 'campus_x': 856, 'campus_y': 492},
    '62': {'name': '생명공학관(62)', 'campus_x': 1022, 'campus_y': 428},
    '70': {'name': '대강당', 'campus_x': 466, 'campus_y': 1546},
    '71': {'name': '의학관', 'campus_x': 682, 'campus_y': 1610},
    '83': {'name': '제2종합연구동', 'campus_x': 1922, 'campus_y': 1278},
    '85': {'name': '산학협력센터', 'campus_x': 1277, 'campus_y': 470},
    '86': {'name': 'N센터', 'campus_x': 1241, 'campus_y': 1687},
}

# ── 도로 길목 노드 (roads.txt 기반) ──────────────────────────────────────────
ROAD_NODES = {
    'R1': (1373, 406), 'R2': (1338, 641), 'R3': (1309, 867), 'R4': (1283, 1046),
    'R5': (1222, 1533), 'R6': (1488, 1584), 'R7': (624, 1457), 'R8': (741, 708),
    'R9': (578, 440), 'R10': (1118, 323), 'R11': (904, 957), 'R12': (1274, 1016),
    'R13': (1604, 1532), 'R14': (1650, 1430), 'R15': (1740, 1418), 'R16': (1845, 1430),
    'R17': (1868, 1275), 'R18': (1712, 1724), 'R19': (1641, 1601), 'R20': (1035, 848),
    'R21': (929, 705), 'R22': (833, 563), 'R23': (1137, 468), 'R24': (1254, 609),
    'R25': (1260, 507), 'R26': (1407, 723), 'R27': (1488, 746), 'R28': (1335, 792),
    'R29': (1329, 1062), 'R30': (1424, 1106), 'R31': (1371, 1193), 'R32': (1265, 1181),
    'R33': (1412, 1377), 'R34': (1386, 1544), 'R35': (695, 1026), 'R36': (609, 1304), 'R37': (650, 1308),
}

ROAD_EDGES = [
    ('R9', 'R10'), ('R1', 'R10'), ('R2', 'R1'), ('R3', 'R2'), ('R12', 'R4'),
    ('R3', 'R12'), ('R4', 'R3'), ('R12', 'R11'), ('R11', 'R8'), ('R8', 'R9'),
    ('R4', 'R5'), ('R7', 'R5'), ('R6', 'R5'), ('R14', 'R15'), ('R16', 'R15'),
    ('R17', 'R16'), ('R14', 'R13'), ('R13', 'R6'), ('R19', 'R13'), ('R19', 'R6'),
    ('R18', 'R19'), ('R8', 'R22'), ('R21', 'R8'), ('R11', 'R20'), ('R23', 'R10'),
    ('R25', 'R23'), ('R24', 'R25'), ('R24', 'R23'), ('R24', 'R2'), ('R26', 'R2'),
    ('R27', 'R26'), ('R3', 'R28'), ('R26', 'R28'), ('R4', 'R29'), ('R29', 'R30'),
    ('R32', 'R4'), ('R32', 'R5'), ('R32', 'R31'), ('R31', 'R33'), ('R33', 'R34'),
    ('R34', 'R6'), ('R34', 'R5'), ('R33', 'R14'), ('R35', 'R8'), ('R11', 'R35'),
    ('R7', 'R35'), ('R35', 'R37'), ('R37', 'R7'), ('R36', 'R37'),
]

# ── 엣지 (건물 간 논리적 연결) ────────────────────────────────────────────────
# 실제 학습은 이 엣지를 기반으로 하되, 시각화만 도로망을 따름
EDGES = [
    ('85', '26', 0.6, 250), ('26', '22', 0.7, 150), ('22', '31', 0.5, 200),
    ('31', '32', 0.8, 80), ('32', '61', 0.5, 180), ('61', '62', 0.9, 40),
    ('22', '33', 0.5, 250), ('33', '40', 0.9, 40), ('21', '22', 0.8, 80),
] # (학습용 엣지는 기존 클러스터 로직 유지 가능)

def build_full_graph():
    """건물 + 도로망을 모두 포함한 큰 그래프 생성 (다익스트라용)"""
    G = nx.Graph()
    # 1. 도로 노드 추가
    for rid, pos in ROAD_NODES.items():
        G.add_node(rid, pos=pos)
    for u, v in ROAD_EDGES:
        p1, p2 = ROAD_NODES[u], ROAD_NODES[v]
        dist = np.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)
        G.add_edge(u, v, weight=dist)
    
    # 2. 건물 노드를 가장 가까운 도로 노드에 연결
    for bid, attr in BUILDINGS.items():
        bx, by = attr['campus_x'], attr['campus_y']
        G.add_node(bid, pos=(bx, by))
        # 가장 가까운 도로 노드 찾기
        best_r, min_d = None, float('inf')
        for rid, rpos in ROAD_NODES.items():
            d = (bx-rpos[0])**2 + (by-rpos[1])**2
            if d < min_d: min_d, best_r = d, rid
        G.add_edge(bid, best_r, weight=np.sqrt(min_d))
    return G

FULL_GRAPH = build_full_graph()

def get_path(start_node, end_node):
    """두 노드 간의 도로 기반 최단 경로 좌표 리스트 반환"""
    try:
        path = nx.shortest_path(FULL_GRAPH, source=start_node, target=end_node, weight='weight')
        return [FULL_GRAPH.nodes[n]['pos'] for n in path]
    except:
        return []

if __name__ == '__main__':
    print(f"도로 노드 {len(ROAD_NODES)}개, 건물 {len(BUILDINGS)}개 통합 완료.")
    p = get_path('85', '21')
    print(f"85동 -> 21동 경로: {len(p)}개 지점 통과")
