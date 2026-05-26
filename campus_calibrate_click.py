"""
campus_calibrate_click.py
지도 위에서 건물 중심을 직접 클릭 → BUILDING_PX 좌표 출력

실행:
  python campus_calibrate_click.py

순서대로 7개 건물을 클릭하면 BUILDING_PX 딕셔너리를 출력해줌.
출력된 좌표를 campus_overlay.py의 BUILDING_PX에 붙여넣으면 됨.
"""
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from PIL import Image

MAP_PATH    = '/home/sean429/swe3032/카카오맵확대.png'
CLICK_ORDER = ['85', '26', '23', '22', '21', '33', '40']
NAMES = {
    '85': 'Cooperation(85)', '26': 'Eng.2(26)', '23': 'Eng.College(23)',
    '22': 'Eng.1(22)',       '21': 'ICT(21)',   '33': 'Pharmacy(33)',
    '40': 'Handok(40)',
}

img = np.array(Image.open(MAP_PATH))
fig, ax = plt.subplots(figsize=(17, 16))
fig.subplots_adjust(top=0.93)
ax.imshow(img)
ax.axis('off')

# 현재 기존 좌표 미리 표시 (회색)
from campus_overlay import BUILDING_PX
for bld, (px, py) in BUILDING_PX.items():
    ax.scatter(px, py, s=200, c='gray', alpha=0.4, zorder=2)
    ax.text(px + 20, py, f'{bld} (old)', fontsize=7, color='gray', alpha=0.6)

collected = {}

def on_click(event):
    if event.inaxes != ax or event.button != 1:
        return
    n = len(collected)
    if n >= len(CLICK_ORDER):
        return
    bld = CLICK_ORDER[n]
    x, y = int(event.xdata), int(event.ydata)
    collected[bld] = (x, y)

    ax.scatter(x, y, s=350, c='red', zorder=5, edgecolors='white', lw=2)
    ax.text(x + 25, y, NAMES[bld], fontsize=10, color='red', fontweight='bold',
            bbox=dict(fc='white', alpha=0.75, ec='none', pad=2))

    remaining = len(CLICK_ORDER) - len(collected)
    if remaining > 0:
        next_bld = CLICK_ORDER[len(collected)]
        ax.set_title(f'Click {NAMES[next_bld]}  ({remaining} left)', fontsize=14, fontweight='bold')
    else:
        ax.set_title('Done! Close the window.', fontsize=14, fontweight='bold', color='green')
        print('\n# ── campus_overlay.py 의 BUILDING_PX를 아래로 교체 ──')
        print('BUILDING_PX = {')
        for k in CLICK_ORDER:
            v = collected[k]
            print(f"    '{k}': {v},  # {NAMES[k]}")
        print('}')
    fig.canvas.draw_idle()

fig.canvas.mpl_connect('button_press_event', on_click)
ax.set_title(f'Click {NAMES[CLICK_ORDER[0]]}  ({len(CLICK_ORDER)} buildings)', fontsize=14, fontweight='bold')
plt.show()
