"""
campus_overlay.py
카카오맵 이미지 위에 건물별 혼잡도(폴리곤 채우기) + 도로망 기반 플로우 오버레이.

사용법:
  python campus_overlay.py
  python campus_overlay.py --day 목 --time 10:30
"""
import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.animation import FuncAnimation, PillowWriter
from PIL import Image

from campus_flow import compute_flows
from campus_graph import get_path, BUILDINGS

# ── 건물별 폴리곤 좌표 (building_places.txt에서 추출됨) ──────────────────
BUILDING_POLY = {
    '05': [(285, 1165), (347, 1177), (357, 1139), (437, 1151), (429, 1185), (591, 1217), (596, 1247), (619, 1260), (612, 1312), (577, 1367), (519, 1391), (256, 1341), (281, 1164)],
    '21': [(1339, 1126), (1555, 1160), (1541, 1205), (1360, 1186), (1329, 1158), (1337, 1124)],
    '22': [(1591, 1026), (1571, 1188), (1633, 1198), (1652, 1041), (1595, 1028)],
    '23': [(1359, 918), (1328, 960), (1325, 1021), (1340, 1040), (1381, 1040), (1421, 976), (1656, 1010), (1659, 980), (1623, 952), (1360, 917)],
    '24': [(1740, 1105), (1860, 1126), (1819, 1400), (1692, 1380), (1739, 1108)],
    '25': [(1403, 788), (1632, 826), (1623, 877), (1416, 848), (1401, 818), (1404, 789)],
    '26': [(1685, 594), (1640, 890), (1681, 900), (1709, 885), (1739, 621), (1723, 601), (1687, 596)],
    '27': [(1491, 566), (1667, 596), (1661, 642), (1489, 620), (1459, 669), (1401, 657), (1417, 545), (1475, 553), (1489, 566)],
    '31': [(1216, 837), (1211, 884), (1104, 972), (947, 944), (1005, 904), (1080, 926), (1215, 836)],
    '32': [(1132, 698), (996, 788), (907, 769), (853, 802), (984, 826), (975, 917), (1009, 900), (1021, 836), (1176, 725), (1228, 753), (1296, 764), (1299, 714), (1184, 694), (1160, 649), (1133, 674), (1131, 704)],
    '33': [(1541, 1773), (1535, 1820), (1675, 1832), (1715, 1816), (1688, 1771), (1547, 1776)],
    '40': [(1691, 1763), (1715, 1803), (1908, 1764), (1877, 1705), (1809, 1716), (1692, 1757), (1693, 1768)],
    '51': [(813, 636), (745, 659), (887, 685), (865, 784), (907, 764), (924, 687), (1037, 603), (1011, 565), (896, 645), (808, 629)],
    '53': [(1389, 1637), (1381, 1661), (1409, 1696), (1487, 1707), (1476, 1785), (1529, 1792), (1536, 1713), (1596, 1716), (1603, 1691), (1583, 1659), (1392, 1635)],
    '61': [(717, 489), (679, 525), (784, 541), (764, 645), (808, 622), (828, 541), (924, 480), (896, 436), (799, 502), (716, 484)],
    '62': [(905, 428), (933, 478), (967, 457), (1108, 470), (1119, 421), (960, 393), (905, 426)],
    '71': [(552, 1523), (531, 1636), (472, 1631), (475, 1708), (564, 1723), (587, 1631), (824, 1679), (828, 1696), (865, 1712), (879, 1699), (887, 1623), (887, 1605), (861, 1573), (767, 1559), (727, 1565), (681, 1557), (649, 1541), (563, 1524)],
    '83': [(1915, 1117), (2047, 1133), (2044, 1310), (2031, 1426), (1872, 1402), (1917, 1120)],
    '85': [(1161, 412), (1151, 464), (1249, 477), (1241, 499), (1283, 527), (1280, 612), (1277, 621), (1228, 615), (1221, 661), (1321, 676), (1348, 435), (1167, 407)],
    '86': [(1205, 1597), (1384, 1627), (1375, 1680), (1264, 1661), (1255, 1737), (1351, 1755), (1340, 1809), (1175, 1792), (1153, 1773), (1175, 1619), (1203, 1623), (1205, 1600)],
}

BUILDING_LABELS = {
    '05': 'Susung', '21': 'ICT', '22': 'Eng.1', '23': 'Eng.College', '24': 'Practice',
    '25': 'Eng.2(25)', '26': 'Eng.2', '27': 'Eng.2(27)', '31': 'Sci.1', '32': 'Sci.2',
    '33': 'Chem.', '40': 'Semicond.', '51': 'Found.', '53': 'Pharm.',
    '61': 'Bio.61', '62': 'Bio.62', '71': 'Med.',
    '83': 'Res.2', '85': 'Coop.', '86': 'N-Center',
}

MAP_PATH   = '/home/sean429/swe3032/maps/카카오맵확대.png'
SNAPSHOT   = {
    '2025-1': '/home/sean429/swe3032/new_snapshot/snapshot_2025_1.csv',
    '2025-2': '/home/sean429/swe3032/new_snapshot/snapshot_2025_2.csv',
}

_DAY_EN = {'월': 'Mon', '화': 'Tue', '수': 'Wed', '목': 'Thu', '금': 'Fri', '토': 'Sat'}


# ── 색상 및 좌표 헬퍼 ──────────────────────────────────────────────────────────
def occupancy_color(occ, max_occ):
    if max_occ == 0 or occ == 0:
        return (0.5, 0.5, 0.5, 0.3)
    ratio = min(occ / max_occ, 1.0)
    if ratio < 0.5:
        r, g = ratio * 2, 1.0
    else:
        r, g = 1.0, (1 - ratio) * 2
    return (r, g, 0.1, 0.7)


def get_poly_center(pts):
    return np.mean(pts, axis=0)


# ── 프레임 드로잉 ─────────────────────────────────────────────────────────────
def draw_frame(ax, img, row, flow_row, max_occ, max_flow, show_zero=False):
    ax.clear()
    ax.imshow(img)
    ax.axis('off')

    day  = _DAY_EN.get(row.get('요일', ''), row.get('요일', ''))
    time = row.get('시각', '')
    ax.set_title(f'{day}  {time}', fontsize=18, fontweight='bold', pad=10, color='white')

    # ── 건물 폴리곤 그리기 ─────────────────────────────────────────────────────
    for bld, pts in BUILDING_POLY.items():
        occ = int(row.get(bld, 0))
        if bld == '62' and '62B08' in row:
            occ += int(row.get('62B08', 0))
            
        color = occupancy_color(occ, max_occ)
        poly = mpatches.Polygon(pts, closed=True, facecolor=color, edgecolor='white', linewidth=1.2, zorder=4)
        ax.add_patch(poly)
        
        cx, cy = get_poly_center(pts)
        if occ > 0 or show_zero:
            label = f"{BUILDING_LABELS.get(bld, bld)}\n{occ}"
            ax.text(cx, cy, label, ha='center', va='center', fontsize=7,
                    fontweight='bold', color='white', zorder=6,
                    bbox=dict(boxstyle='round,pad=0.15', fc='#000000aa', ec='none'))

    # ── 도로망 기반 플로우 그리기 ───────────────────────────────────────────────
    for _, fr in flow_row.iterrows():
        src, dst, flow = fr['from'], fr['to'], fr['flow']
        if flow <= 0: continue

        path_pts = get_path(src, dst)
        if len(path_pts) < 2: continue

        # 경로 그리기
        path_pts = np.array(path_pts)
        lw = max(1.5, min(12, flow / 15))
        
        # 도로를 따라가는 선
        ax.plot(path_pts[:, 0], path_pts[:, 1], color='#4fc3f7cc', lw=lw, zorder=5)
        
        # 마지막 구간에 화살표 표시
        p_last2, p_last = path_pts[-2], path_pts[-1]
        ax.annotate('', xy=(p_last[0], p_last[1]), xytext=(p_last2[0], p_last2[1]),
            arrowprops=dict(arrowstyle='->', color='#4fc3f7cc', lw=lw, mutation_scale=15),
            zorder=5)

    # ── 범례 ──────────────────────────────────────────────────────────────────
    occ_legend = [
        mpatches.Patch(color=occupancy_color(0,   1), label='Empty'),
        mpatches.Patch(color=occupancy_color(0.3, 1), label='Low'),
        mpatches.Patch(color=occupancy_color(0.6, 1), label='Medium'),
        mpatches.Patch(color=occupancy_color(1.0, 1), label='High'),
    ]
    ax.legend(handles=occ_legend, loc='lower left', fontsize=9, framealpha=0.8, title='Occupancy')


def make_gif(day: str, out_path: str, semester: str = '2025-1'):
    df = pd.read_csv(SNAPSHOT[semester])
    rows = df[df['요일'] == day].reset_index(drop=True)
    if rows.empty: return

    flows = compute_flows(df)
    day_flow = flows[flows['요일'] == day]
    
    df_occ = df.copy()
    if '62B08' in df_occ.columns and '62' in df_occ.columns:
        df_occ['62'] = df_occ['62'] + df_occ['62B08']
        
    max_occ = max(int(df_occ[b].max()) for b in BUILDING_POLY.keys() if b in df_occ.columns)
    max_flow = day_flow['flow'].max() if not day_flow.empty else 1
    img = np.array(Image.open(MAP_PATH))

    fig, ax = plt.subplots(figsize=(14, 13))
    fig.patch.set_facecolor('#1a1a2e')

    def update(i):
        t = rows.iloc[i]['시각']
        flow_row = day_flow[day_flow['시각'] == t]
        draw_frame(ax, img, rows.iloc[i].to_dict(), flow_row, max_occ, max_flow)

    ani = FuncAnimation(fig, update, frames=len(rows), interval=500)
    ani.save(out_path, writer=PillowWriter(fps=2))
    plt.close()
    print(f'Saved GIF: {out_path}')

def make_png(day: str, time: str, out_path: str, semester: str = '2025-1'):
    df = pd.read_csv(SNAPSHOT[semester])
    rows = df[(df['요일'] == day) & (df['시각'] == time)]
    if rows.empty: return

    flows = compute_flows(df)
    flow_row = flows[(flows['요일'] == day) & (flows['시각'] == time)]
    
    df_occ = df.copy()
    if '62B08' in df_occ.columns and '62' in df_occ.columns:
        df_occ['62'] = df_occ['62'] + df_occ['62B08']

    max_occ = max(int(df_occ[b].max()) for b in BUILDING_POLY.keys() if b in df_occ.columns)
    max_flow = flows['flow'].max() if not flows.empty else 1
    img = np.array(Image.open(MAP_PATH))

    fig, ax = plt.subplots(figsize=(14, 13))
    fig.patch.set_facecolor('#1a1a2e')
    draw_frame(ax, img, rows.iloc[0].to_dict(), flow_row, max_occ, max_flow, show_zero=True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=120, bbox_inches='tight')
    plt.close()
    print(f'Saved PNG: {out_path}')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--day',      default=None)
    parser.add_argument('--time',     default=None)
    parser.add_argument('--semester', default='2025-1', choices=['2025-1', '2025-2'])
    args = parser.parse_args()

    DAYS = ['월', '화', '수', '목', '금']
    sem = args.semester.replace('-', '_')

    if args.time:
        day = args.day or '월'
        out = f'animations/campus_{sem}_{day}_{args.time.replace(":","-")}.png'
        make_png(day, args.time, out, args.semester)
    else:
        days = [args.day] if args.day else DAYS
        for day in days:
            out = f'animations/campus_{sem}_{day}.gif'
            make_gif(day, out, args.semester)
