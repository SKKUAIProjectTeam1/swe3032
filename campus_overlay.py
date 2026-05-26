"""
campus_overlay.py
카카오맵 이미지 위에 건물별 혼잡도(체류 인원) + 건물 간 이동 플로우 오버레이.

사용법:
  python campus_overlay.py                      # 월요일 전체 GIF
  python campus_overlay.py --day 목 --time 10:30 # 특정 시각 PNG
  python campus_overlay.py --calibrate           # 좌표 보정용 마커 이미지
"""
import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.patheffects as pe
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.patches import FancyArrowPatch
from PIL import Image

from campus_flow import compute_flows

# ── 건물별 픽셀 좌표 (2223×2056 기준) ────────────────────────────────────────
BUILDING_PX = {
    '85': (1288, 465),   # 산학협력센터
    '26': (1693, 721),   # 제2공학관
    '23': (1517, 966),   # 공과대학
    '22': (1609, 1111),  # 제1공학관
    '21': (1457, 1170),  # 정보통신대학
    '33': (1613, 1800),  # 화학관
    '40': (1812, 1749),  # 반도체관
}

BUILDING_LABELS = {
    '85': '85\nCooperation',
    '26': '26\nEng.2',
    '23': '23\nEng.College',
    '22': '22\nEng.1',
    '21': '21\nICT',
    '33': '33\nChem.',
    '40': '40\nSemicond.',
}

MAP_PATH      = '/home/sean429/swe3032/카카오맵확대.png'
SNAPSHOT_PATH = '/home/sean429/swe3032/2025_1_snapshot.csv'
BUILDINGS     = list(BUILDING_PX.keys())

_DAY_EN = {'월': 'Mon', '화': 'Tue', '수': 'Wed', '목': 'Thu', '금': 'Fri', '토': 'Sat'}


# ── 색상 헬퍼 ─────────────────────────────────────────────────────────────────
def occupancy_color(occ, max_occ):
    if max_occ == 0:
        return (0.3, 0.8, 0.3, 0.75)
    ratio = min(occ / max_occ, 1.0)
    r = min(1.0, ratio * 2)
    g = min(1.0, (1 - ratio) * 2)
    return (r, g, 0.1, 0.78)


def flow_color(flow, max_flow):
    if max_flow == 0:
        return '#4fc3f7'
    ratio = min(flow / max_flow, 1.0)
    # 하늘색(낮음) → 주황(높음)
    r = int(79  + (255 - 79)  * ratio)
    g = int(195 + (140 - 195) * ratio)
    b = int(247 + (0   - 247) * ratio)
    return f'#{r:02x}{g:02x}{b:02x}'


# ── 프레임 드로잉 ─────────────────────────────────────────────────────────────
def draw_frame(ax, img, row, flow_row, max_occ, max_flow, show_zero=False):
    ax.clear()
    ax.imshow(img)
    ax.axis('off')

    day  = _DAY_EN.get(row.get('요일', ''), row.get('요일', ''))
    time = row.get('시각', '')
    ax.set_title(f'{day}  {time}', fontsize=16, fontweight='bold', pad=8)

    # ── 엣지 플로우 ────────────────────────────────────────────────────────────
    drawn = set()
    for _, fr in flow_row.iterrows():
        src, dst, flow = fr['from'], fr['to'], fr['flow']
        if src not in BUILDING_PX or dst not in BUILDING_PX or flow <= 0:
            continue

        x0, y0 = BUILDING_PX[src]
        x1, y1 = BUILDING_PX[dst]
        pair    = (src, dst)
        rad     = 0.25 if (dst, src) in drawn else 0.0
        drawn.add(pair)

        lw    = max(1.5, min(14, flow / 18))
        color = flow_color(flow, max_flow)

        ax.annotate('', xy=(x1, y1), xytext=(x0, y0),
            arrowprops=dict(
                arrowstyle='->', color=color, lw=lw,
                connectionstyle=f'arc3,rad={rad}',
                mutation_scale=18,
            ), zorder=2)

        # 플로우 수치 레이블
        mx = (x0 + x1) / 2 + (y1 - y0) * 0.08 * rad
        my = (y0 + y1) / 2 - (x1 - x0) * 0.08 * rad
        ax.text(mx, my, f'{int(flow)}',
                fontsize=7, fontweight='bold', color='white', ha='center', va='center',
                bbox=dict(boxstyle='round,pad=0.15', fc='#00000099', ec='none'),
                zorder=3)

    # ── 건물 노드 ──────────────────────────────────────────────────────────────
    for bld, (px, py) in BUILDING_PX.items():
        occ = int(row.get(bld, 0))
        if occ == 0 and not show_zero:
            ax.scatter(px, py, s=80, c='#cccccc', alpha=0.5, zorder=4)
            continue

        radius_pt = max(60, min(2200, occ * 0.9))
        color     = occupancy_color(occ, max_occ)
        ax.scatter(px, py, s=radius_pt, c=[color], zorder=5,
                   edgecolors='white', linewidths=1.5)
        ax.text(px, py - 42, BUILDING_LABELS[bld],
                ha='center', va='bottom', fontsize=7.5,
                fontweight='bold', color='white', zorder=6,
                bbox=dict(boxstyle='round,pad=0.2', fc='#00000088', ec='none'))
        ax.text(px, py + 42, f'{occ}',
                ha='center', va='top', fontsize=8,
                fontweight='bold', color='white', zorder=6,
                bbox=dict(boxstyle='round,pad=0.2', fc='#00000088', ec='none'))

    # ── 범례 ──────────────────────────────────────────────────────────────────
    occ_legend = [
        mpatches.Patch(color=occupancy_color(0,   1), label='Empty'),
        mpatches.Patch(color=occupancy_color(0.3, 1), label='Low'),
        mpatches.Patch(color=occupancy_color(0.6, 1), label='Medium'),
        mpatches.Patch(color=occupancy_color(1.0, 1), label='High'),
    ]
    flow_legend = [
        mpatches.Patch(color=flow_color(0,   1), label='Flow: low'),
        mpatches.Patch(color=flow_color(1.0, 1), label='Flow: high'),
    ]
    leg1 = ax.legend(handles=occ_legend,  loc='lower left',  fontsize=8.5,
                     framealpha=0.88, labelcolor='black', title='Occupancy')
    ax.add_artist(leg1)
    ax.legend(handles=flow_legend, loc='lower right', fontsize=8.5,
              framealpha=0.88, labelcolor='black', title='Movement')


# ── 보정용 ────────────────────────────────────────────────────────────────────
def calibrate():
    img = np.array(Image.open(MAP_PATH))
    fig, ax = plt.subplots(figsize=(14, 13))
    ax.imshow(img)
    ax.axis('off')
    ax.set_title('Calibration — adjust BUILDING_PX if markers are off', fontsize=13)
    for bld, (px, py) in BUILDING_PX.items():
        ax.scatter(px, py, s=400, c='red', zorder=5, edgecolors='white', lw=2)
        ax.text(px + 30, py, f'{bld}  ({px},{py})', fontsize=11,
                color='red', fontweight='bold',
                bbox=dict(fc='white', alpha=0.7, ec='none', pad=2))
    out = '/home/sean429/swe3032/campus_calibrate.png'
    plt.tight_layout()
    plt.savefig(out, dpi=100, bbox_inches='tight')
    print(f'Saved: {out}')


# ── GIF / PNG ─────────────────────────────────────────────────────────────────
def make_gif(day: str, out_path: str):
    df    = pd.read_csv(SNAPSHOT_PATH)
    rows  = df[df['요일'] == day].reset_index(drop=True)
    if rows.empty:
        print(f'[ERROR] {day} 데이터 없음'); return

    flows    = compute_flows(df)
    day_flow = flows[flows['요일'] == day]

    max_occ  = max(int(rows[b].max())        for b in BUILDINGS if b in rows.columns)
    max_flow = day_flow['flow'].max() if not day_flow.empty else 1
    img      = np.array(Image.open(MAP_PATH))

    fig, ax = plt.subplots(figsize=(14, 13))
    fig.patch.set_facecolor('#111')

    def update(i):
        t        = rows.iloc[i]['시각']
        flow_row = day_flow[day_flow['시각'] == t]
        draw_frame(ax, img, rows.iloc[i].to_dict(), flow_row, max_occ, max_flow)

    ani = FuncAnimation(fig, update, frames=len(rows), interval=600)
    ani.save(out_path, writer=PillowWriter(fps=2))
    plt.close()
    print(f'Saved GIF: {out_path}  ({len(rows)} frames)')


def make_png(day: str, time: str, out_path: str):
    df   = pd.read_csv(SNAPSHOT_PATH)
    rows = df[(df['요일'] == day) & (df['시각'] == time)]
    if rows.empty:
        print(f'[ERROR] {day} {time} 데이터 없음'); return

    flows    = compute_flows(df)
    flow_row = flows[(flows['요일'] == day) & (flows['시각'] == time)]

    max_occ  = max(int(df[b].max())   for b in BUILDINGS if b in df.columns)
    max_flow = flows['flow'].max() if not flows.empty else 1
    img      = np.array(Image.open(MAP_PATH))

    fig, ax = plt.subplots(figsize=(14, 13))
    fig.patch.set_facecolor('#111')
    draw_frame(ax, img, rows.iloc[0].to_dict(), flow_row, max_occ, max_flow, show_zero=True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=120, bbox_inches='tight')
    plt.close()
    print(f'Saved PNG: {out_path}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--calibrate', action='store_true')
    parser.add_argument('--day',  default='월')
    parser.add_argument('--time', default=None)
    args = parser.parse_args()

    if args.calibrate:
        calibrate()
    elif args.time:
        make_png(args.day, args.time,
                 f'/home/sean429/swe3032/campus_{args.day}_{args.time.replace(":","-")}.png')
    else:
        make_gif(args.day,
                 f'/home/sean429/swe3032/campus_{args.day}.gif')
