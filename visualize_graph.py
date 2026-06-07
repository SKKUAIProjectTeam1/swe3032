"""
visualize_graph.py
캠퍼스 건물 그래프 구조 시각화 (현재 정의된 노드/엣지 확인)
"""
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from campus_graph import BUILDINGS, EDGES

def visualize():
    G = nx.DiGraph()
    
    # 노드 추가
    pos = {}
    labels = {}
    for bld, attr in BUILDINGS.items():
        G.add_node(bld)
        # 이미지 좌표계(Y증가=아래)를 수학 좌표계(Y증가=위)로 보정하기 위해 Y값에 -를 붙임
        pos[bld] = np.array([attr['campus_x'], -attr['campus_y']])
        labels[bld] = bld

    # 엣지 추가
    for u, v, w, dist in EDGES:
        G.add_edge(u, v, weight=w)

    plt.figure(figsize=(12, 10))
    plt.gca().set_facecolor('#f9f9f9')

    # 노드 그리기
    nx.draw_networkx_nodes(G, pos, node_size=800, node_color='skyblue', 
                           edgecolors='white', linewidths=2, alpha=0.9)

    
    # 엣지 그리기 (가중치에 따라 두께 조절)
    weights = [G[u][v]['weight'] * 3 for u, v in G.edges()]
    nx.draw_networkx_edges(G, pos, width=weights, edge_color='gray', 
                           arrows=True, arrowsize=15, connectionstyle='arc3,rad=0.1')
    
    # 레이블
    nx.draw_networkx_labels(G, pos, labels, font_size=10, font_weight='bold')

    plt.title("Campus Building Graph Structure (Redefined Edges)", fontsize=15)
    plt.axis('off')
    
    out_path = 'plots/campus_graph_structure.png'
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    print(f"Saved graph visualization to: {out_path}")
    plt.close()

if __name__ == '__main__':
    visualize()
