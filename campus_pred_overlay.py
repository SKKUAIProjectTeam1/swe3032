"""
campus_pred_overlay.py
예측 vs 실제 혼잡도 비교 GIF — 카카오맵 위 나란히 표시.

사용법:
  python campus_pred_overlay.py                        # 전 요일 × 전 학기
  python campus_pred_overlay.py --semester 2025-1      # 1학기만
  python campus_pred_overlay.py --semester 2025-1 --day 화
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

from campus_overlay import BUILDING_PX, BUILDING_LABELS, MAP_PATH, occupancy_color

PRED_CSV = {
    '2025-1': '/home/sean429/swe3032/results/2025_1_pred.csv',
    '2025-2': '/home/sean429/swe3032/results/2025_2_pred.csv',
}
OUT_PATH = '/home/sean429/swe3032/animations/campus_pred_{sem}_{day}.gif'
DAYS     = ['월', '화', '수', '목', '금']
_DAY_EN  = {'월': 'Mon', '화': 'Tue', '수': 'Wed', '목': 'Thu', '금': 'Fri', '토': 'Sat'}
BUILDINGS = list(BUILDING_PX.keys())


def _draw_panel(ax, img, occ_dict: dict, max_occ: float, title: str):
    ax.clear()
    ax.imshow(img)
    ax.axis('off')
    ax.set_title(title, fontsize=13, fontweight='bold', pad=6, color='white')
    ax.set_facecolor('#111')

    for bld, (px, py) in BUILDING_PX.items():
        occ = max(0.0, occ_dict.get(bld, 0.0))
        if occ < 1:
            ax.scatter(px, py, s=80, c='#aaaaaa', alpha=0.4, zorder=4)
            continue
        radius_pt = max(60, min(2200, occ * 0.9))
        color = occupancy_color(occ, max_occ)
        ax.scatter(px, py, s=radius_pt, c=[color], zorder=5,
                   edgecolors='white', linewidths=1.5)
        ax.text(px, py - 42, BUILDING_LABELS[bld],
                ha='center', va='bottom', fontsize=7.5, fontweight='bold',
                color='white', zorder=6,
                bbox=dict(boxstyle='round,pad=0.2', fc='#00000088', ec='none'))
        ax.text(px, py + 42, f'{int(occ)}',
                ha='center', va='top', fontsize=8, fontweight='bold',
                color='white', zorder=6,
                bbox=dict(boxstyle='round,pad=0.2', fc='#00000088', ec='none'))

    occ_legend = [
        mpatches.Patch(color=occupancy_color(0,   1), label='Empty'),
        mpatches.Patch(color=occupancy_color(0.3, 1), label='Low'),
        mpatches.Patch(color=occupancy_color(0.7, 1), label='Medium'),
        mpatches.Patch(color=occupancy_color(1.0, 1), label='High'),
    ]
    ax.legend(handles=occ_legend, loc='lower left', fontsize=8, framealpha=0.85)


def make_comparison_gif(semester: str, day: str):
    df = pd.read_csv(PRED_CSV[semester])
    df['building'] = df['building'].astype(str)
    df_day = df[df['요일'] == day].copy()
    if df_day.empty:
        print(f'[SKIP] {semester} {day} 데이터 없음')
        return

    # wide format: 시각 × building → actual / pred_MLP
    times = sorted(df_day['시각'].unique())
    max_occ = max(df_day['actual'].max(), df_day['pred_MLP'].max())

    img = np.array(Image.open(MAP_PATH))
    fig, axes = plt.subplots(1, 2, figsize=(26, 13))
    fig.patch.set_facecolor('#111')
    fig.subplots_adjust(wspace=0.03)

    def update(i):
        t    = times[i]
        slot = df_day[df_day['시각'] == t]
        actual = {r['building']: r['actual']   for _, r in slot.iterrows()}
        pred   = {r['building']: r['pred_MLP'] for _, r in slot.iterrows()}
        day_en = _DAY_EN.get(day, day)
        _draw_panel(axes[0], img, actual, max_occ, f'Actual  |  {day_en} {t}')
        _draw_panel(axes[1], img, pred,   max_occ, f'MLP Predicted  |  {day_en} {t}')

    ani = FuncAnimation(fig, update, frames=len(times), interval=700)
    sem = semester.replace('-', '_')
    out = OUT_PATH.format(sem=sem, day=day)
    ani.save(out, writer=PillowWriter(fps=1.5))
    plt.close()
    print(f'Saved: {out}  ({len(times)} frames)')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--semester', default=None, choices=['2025-1', '2025-2'])
    parser.add_argument('--day',      default=None)
    args = parser.parse_args()

    semesters = [args.semester] if args.semester else ['2025-1', '2025-2']
    days      = [args.day]      if args.day      else DAYS

    for sem in semesters:
        for day in days:
            make_comparison_gif(sem, day)
