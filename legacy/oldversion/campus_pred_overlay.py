"""
campus_pred_overlay.py
예측 vs 실제 혼잡도 비교 GIF (폴리곤 방식) — 카카오맵 위 나란히 표시.

사용법:
  python campus_pred_overlay.py                        # 전 요일 × 전 학기
  python campus_pred_overlay.py --semester 2025-2      # 2학기만
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

# campus_overlay에서 폴리곤 데이터와 유틸 가져오기
from campus_overlay import BUILDING_POLY, BUILDING_LABELS, MAP_PATH, occupancy_color, get_poly_center

PRED_CSV = {
    '2025-1': '/home/sean429/swe3032/results/2025_1_pred.csv',
    '2025-2': '/home/sean429/swe3032/results/2025_2_pred.csv',
}
OUT_PATH = '/home/sean429/swe3032/animations/campus_pred_{sem}_{day}.gif'
DAYS     = ['월', '화', '수', '목', '금']
_DAY_EN  = {'월': 'Mon', '화': 'Tue', '수': 'Wed', '목': 'Thu', '금': 'Fri', '토': 'Sat'}


def _draw_panel(ax, img, occ_dict: dict, max_occ: float, title: str):
    ax.clear()
    ax.imshow(img)
    ax.axis('off')
    ax.set_title(title, fontsize=18, fontweight='bold', pad=10, color='white')

    for bld, pts in BUILDING_POLY.items():
        occ = max(0.0, occ_dict.get(bld, 0.0))
        color = occupancy_color(occ, max_occ)
        
        # 폴리곤 그리기
        poly = mpatches.Polygon(pts, closed=True, facecolor=color, edgecolor='white', linewidth=1.2, zorder=4)
        ax.add_patch(poly)
        
        # 중심점 계산 (레이블용)
        cx, cy = get_poly_center(pts)
        
        # 인원수 텍스트 (값 이 있을 때만)
        if occ >= 1:
            label = f"{BUILDING_LABELS.get(bld, bld)}\n{int(occ)}"
            ax.text(cx, cy, label, ha='center', va='center', fontsize=7,
                    fontweight='bold', color='white', zorder=6,
                    bbox=dict(boxstyle='round,pad=0.2', fc='#00000088', ec='none'))

    occ_legend = [
        mpatches.Patch(color=occupancy_color(0,   1), label='Empty'),
        mpatches.Patch(color=occupancy_color(0.3, 1), label='Low'),
        mpatches.Patch(color=occupancy_color(0.6, 1), label='Medium'),
        mpatches.Patch(color=occupancy_color(1.0, 1), label='High'),
    ]
    ax.legend(handles=occ_legend, loc='lower left', fontsize=10, framealpha=0.8, title='Occupancy')


def make_comparison_gif(semester: str, day: str):
    csv_path = PRED_CSV[semester]
    import os
    if not os.path.exists(csv_path):
        print(f'[SKIP] {csv_path} 파일 없음 (먼저 campus_gnn.py 실행 필요)')
        return

    df = pd.read_csv(csv_path)
    df['building'] = df['building'].astype(str)
    df_day = df[df['요일'] == day].copy()
    if df_day.empty:
        print(f'[SKIP] {semester} {day} 데이터 없음')
        return

    times = sorted(df_day['시각'].unique())
    # MLP 예측값 기준 시각화 (GAT로 변경 가능)
    max_occ = max(df_day['actual'].max(), df_day['pred_MLP'].max())

    img = np.array(Image.open(MAP_PATH))
    fig, axes = plt.subplots(1, 2, figsize=(26, 13))
    fig.patch.set_facecolor('#1a1a2e')
    fig.subplots_adjust(wspace=0.01, left=0.02, right=0.98)

    def update(i):
        t    = times[i]
        slot = df_day[df_day['시각'] == t]
        actual = {r['building']: r['actual']   for _, r in slot.iterrows()}
        pred   = {r['building']: r['pred_MLP'] for _, r in slot.iterrows()}
        day_en = _DAY_EN.get(day, day)
        _draw_panel(axes[0], img, actual, max_occ, f'ACTUAL  |  {day_en} {t}')
        _draw_panel(axes[1], img, pred,   max_occ, f'PREDICTED (MLP)  |  {day_en} {t}')

    ani = FuncAnimation(fig, update, frames=len(times), interval=600)
    sem = semester.replace('-', '_')
    out = OUT_PATH.format(sem=sem, day=day)
    ani.save(out, writer=PillowWriter(fps=2))
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
