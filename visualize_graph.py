import sys
sys.path.insert(0, '/home/sean429/swe3032')
import networkx as nx
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from campus_graph import BUILDINGS, EDGES

ENG_NAMES = {
    '21': 'ICT College', '22': 'Eng.Bldg-1', '23': 'Eng.College',
    '26': 'Eng.Bldg-2\n(cluster gate)', '33': 'Chem.Bldg',
    '40': 'Semicond.', '85': 'Cooperation\nCenter',
}

G = nx.DiGraph()
for bld_id, attrs in BUILDINGS.items():
    G.add_node(bld_id, **attrs)
for src, dst, w, dist in EDGES:
    G.add_edge(src, dst, entrance_weight=w, distance=dist)

pos = {b: (attr['campus_x'], attr['campus_y']) for b, attr in BUILDINGS.items()}
fig, ax = plt.subplots(figsize=(12, 11))
ax.set_facecolor('#f0f4f8')
fig.patch.set_facecolor('#f0f4f8')

# cluster backgrounds
# Hub cluster: 21(105,-245) / 22(210,-220) / 23(110,-165) — all branch from 22동
ax.add_patch(plt.Polygon([(55,-280),(265,-280),(265,-130),(55,-130)],
    closed=True, fill=True, facecolor='#cce5ff', edgecolor='#4a90d9', lw=1.5, alpha=0.35, zorder=0))
# 25/26/27 cluster: 26(220,-100) is the gate
ax.add_patch(plt.Polygon([(160,-125),(275,-125),(275,-55),(160,-55)],
    closed=True, fill=True, facecolor='#d4edda', edgecolor='#28a745', lw=1.5, alpha=0.35, zorder=0))
# Complex 33/40: 33(185,-430) / 40(250,-455)
ax.add_patch(plt.Polygon([(140,-400),(300,-400),(300,-490),(140,-490)],
    closed=True, fill=True, facecolor='#f8d7da', edgecolor='#c0392b', lw=1.5, alpha=0.35, zorder=0))
ax.text(160, -270, 'Hub Cluster 21/22/23', ha='center', fontsize=8.5, color='#1a6bb5', fontweight='bold')
ax.text(218,  -65, 'Cluster 25/26/27',     ha='center', fontsize=8.5, color='#155724', fontweight='bold')
ax.text(220, -480, 'Complex 33/40',         ha='center', fontsize=8.5, color='#922b21', fontweight='bold')

# parking lot 3 (22↔33 경유지)
ax.add_patch(mpatches.FancyBboxPatch((148, -395), 112, 90, boxstyle='round,pad=3',
    fill=True, facecolor='#e0e0e0', edgecolor='#999', lw=1, alpha=0.7, zorder=0))
ax.text(204, -345, '[Parking Lot 3]', ha='center', fontsize=8, color='#777', style='italic')

# edges
drawn = set()
for src, dst, data in G.edges(data=True):
    w = data['entrance_weight']
    dist = data['distance']
    pair = tuple(sorted([src, dst]))
    rad = 0.13 if pair in drawn else 0.0
    drawn.add(pair)
    color = '#1565C0' if w >= 0.5 else '#64B5F6'
    x0, y0 = pos[src]; x1, y1 = pos[dst]
    ax.annotate('', xy=(x1, y1), xytext=(x0, y0),
        arrowprops=dict(arrowstyle='->', color=color, lw=w * 5.5,
                        connectionstyle=f'arc3,rad={rad}'))
    mx, my = (x0 + x1) / 2 + 12, (y0 + y1) / 2
    ax.text(mx, my, f'w={w}\n{dist}m', fontsize=6.5, color='#333',
            ha='left', va='center', linespacing=1.3)

# nodes
colors = {'21':'#E65100','22':'#EF6C00','23':'#F57C00',
          '26':'#2E7D32','33':'#6A1B9A','40':'#7B1FA2','85':'#B71C1C'}
sizes  = {'85':2800,'26':2200,'22':2000,'23':2000,'21':1700,'33':1700,'40':1500}
offsets = {'21':(-52,16),'22':(-52,-22),'23':(-55,16),
           '26':(-62,16),'85':(18,16),'33':(-52,16),'40':(18,16)}

for node, (x, y) in pos.items():
    ax.scatter(x, y, s=sizes[node], c=colors[node],
               zorder=5, edgecolors='white', linewidths=2.5, alpha=0.93)
    ax.text(x, y, node, ha='center', va='center',
            fontsize=12, fontweight='bold', color='white', zorder=6)
    dx, dy = offsets[node]
    ax.text(x + dx, y + dy, ENG_NAMES[node], fontsize=8, color='#111',
            ha='center', fontweight='bold', linespacing=1.2)

legend = [
    mpatches.Patch(color='#B71C1C', label='85  Cooperation Center  (N anchor)'),
    mpatches.Patch(color='#2E7D32', label='26  Eng.Bldg-2  (cluster gate)'),
    mpatches.Patch(color='#F57C00', label='23  Eng.College  (main entry)'),
    mpatches.Patch(color='#EF6C00', label='22  Eng.Bldg-1  (hub)'),
    mpatches.Patch(color='#E65100', label='21  ICT College'),
    mpatches.Patch(color='#6A1B9A', label='33/40  Chem./Semicond.  (S of parking)'),
    mpatches.Patch(color='#1565C0', label='Main path  (w >= 0.5)'),
    mpatches.Patch(color='#64B5F6', label='Secondary  (w < 0.5)'),
]
ax.legend(handles=legend, loc='upper left', fontsize=8.5, framealpha=0.9)
ax.set_title('SKKU Natural Science Campus — Building Movement Graph\n'
             '(Arrow width = entrance weight  |  Label = walking distance)',
             fontsize=13, pad=14)
ax.set_xlabel('<- West                                           East ->', fontsize=9)
ax.set_ylabel('<- South                              North ->', fontsize=9)
ax.grid(True, alpha=0.2, linestyle='--')
ax.set_xlim(-80, 330)
ax.set_ylim(-580, 70)
plt.tight_layout()
plt.savefig('/home/sean429/swe3032/plots/campus_graph.png', dpi=150, bbox_inches='tight')
print('Saved: campus_graph.png')
