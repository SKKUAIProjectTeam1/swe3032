"""
campus_eval_plot.py
2025_2_pred.csv 읽어서 예측 vs 실제 시각화 (폰트 재설정 포함).

사용법:
  python campus_eval_plot.py              # 전 요일 PNG 저장
  python campus_eval_plot.py --day 화     # 특정 요일만
"""
import argparse
import sys
sys.path.insert(0, '/home/sean429/swe3032')

import matplotlib
matplotlib.use('Agg')
import koreanize_matplotlib  # noqa: registers NanumGothic automatically
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from campus_graph import BUILDINGS

BUILDING_IDS = list(BUILDINGS.keys())
_BLDG_LABEL  = {b: BUILDINGS[b]['name'] for b in BUILDING_IDS}
_DAY_EN      = {'월': 'Mon', '화': 'Tue', '수': 'Wed', '목': 'Thu', '금': 'Fri', '토': 'Sat'}

PRED_CSV  = '/home/sean429/swe3032/2025_2_pred.csv'
OUT_PLOT  = '/home/sean429/swe3032/campus_eval_{day}.png'

MODEL_STYLE = {
    'MLP': dict(color='#3498db', ls='--', marker='s', lw=1.8, ms=4),
    'GCN': dict(color='#e67e22', ls='--', marker='^', lw=1.8, ms=4),
    'GAT': dict(color='#e74c3c', ls='-',  marker='o', lw=2.2, ms=5),
}


def setup_korean_font():
    """koreanize_matplotlib import 시 이미 NanumGothic 등록됨."""
    matplotlib.rcParams['axes.unicode_minus'] = False
    print(f'[font] {matplotlib.rcParams["font.family"]}')
    return True


def plot_day(df_comp: pd.DataFrame, day: str, out_path: str, use_korean: bool):
    sub = df_comp[df_comp['요일'] == day]
    if sub.empty:
        print(f'[SKIP] {day} 데이터 없음')
        return

    n_bld  = len(BUILDING_IDS)
    n_cols = 4
    n_rows = (n_bld + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5.5 * n_cols, 4.5 * n_rows))
    axes = axes.flatten()

    for idx, b in enumerate(BUILDING_IDS):
        ax   = axes[idx]
        bsub = sub[sub['building'] == b].sort_values('시각')

        label_bld = _BLDG_LABEL[b] if use_korean else f'Bldg {b}'

        if bsub.empty or bsub['actual'].max() == 0:
            ax.text(0.5, 0.5, 'No class', ha='center', va='center',
                    transform=ax.transAxes, fontsize=12, color='#999')
            ax.set_title(f'{b}  {label_bld}', fontsize=10)
            ax.axis('off')
            continue

        times = bsub['시각'].values
        x_pos = np.arange(len(times))
        actual = bsub['actual'].values

        ax.fill_between(x_pos, actual, alpha=0.10, color='black')
        ax.plot(x_pos, actual, 'k-o', lw=2.2, ms=5, label='Actual', zorder=5)

        for name, style in MODEL_STYLE.items():
            col = f'pred_{name}'
            if col in bsub.columns:
                ax.plot(x_pos, bsub[col].values, label=name, zorder=4, **style)

        step = max(1, len(times) // 6)
        ax.set_xticks(x_pos[::step])
        ax.set_xticklabels(times[::step], rotation=45, fontsize=7.5)
        ax.set_ylabel('명 (persons)', fontsize=8)
        peak = actual.max()
        ax.set_title(f'{b}  {label_bld}  (peak {peak:.0f})', fontsize=9.5)
        ax.legend(fontsize=7.5, loc='upper right', framealpha=0.75)
        ax.grid(alpha=0.3)
        ax.set_xlim(-0.5, len(times) - 0.5)
        ax.set_ylim(bottom=0)

    for ax in axes[n_bld:]:
        ax.axis('off')

    day_en  = _DAY_EN.get(day, day)
    day_str = f'{day}요일 ({day_en})' if use_korean else day_en

    fig.suptitle(
        f'2025-2  {day_str}  —  Building Congestion: Predicted vs Actual\n'
        f'Train: 2025-1 synthetic  |  Test: 2025-2 ground truth',
        fontsize=13, fontweight='bold', y=1.01
    )
    plt.tight_layout()
    plt.savefig(out_path, dpi=130, bbox_inches='tight')
    plt.close()
    print(f'Saved: {out_path}')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--day', default=None, help='요일 (미지정 시 전체)')
    args = parser.parse_args()

    use_korean = setup_korean_font()

    df = pd.read_csv(PRED_CSV)
    df['building'] = df['building'].astype(str)
    days = [args.day] if args.day else sorted(df['요일'].unique(),
            key=lambda d: ['월','화','수','목','금','토'].index(d) if d in ['월','화','수','목','금','토'] else 9)

    for day in days:
        out = OUT_PLOT.format(day=day)
        plot_day(df, day, out, use_korean)


if __name__ == '__main__':
    main()
