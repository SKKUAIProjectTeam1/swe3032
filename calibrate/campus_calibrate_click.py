"""
campus_calibrate_click.py (Full Version)
지도 위에서 모든 건물의 중심점(Centroid)을 클릭하여 좌표를 수집합니다.
"""
import numpy as np
import matplotlib
try:
    matplotlib.use('TkAgg')
except:
    matplotlib.use('Agg') # GUI가 안되는 환경 대비
import matplotlib.pyplot as plt
from PIL import Image
import os

MAP_PATH = '/home/sean429/swe3032/maps/카카오맵확대.png'

# 전체 건물 목록 (순서대로 클릭하게 됩니다)
CLICK_ORDER = [
    '03', '05', '21', '22', '23', '24', '25', '26', '27', '31', 
    '32', '33', '40', '48', '51', '53', '61', '62', '70', '71', 
    '83', '85', '86'
]

NAMES = {
    '03': '학생회관', '05': '수성관', '21': '정보통신대학', '22': '제1공학관',
    '23': '공과대학', '24': '공학실습동', '25': '제2공학관(25)', '26': '제2공학관(26)',
    '27': '제2공학관(27)', '31': '제1과학관', '32': '제2과학관', '33': '화학관',
    '40': '반도체관', '48': '삼성학술정보관', '51': '기초학문관', '53': '약학관',
    '61': '생명공학관(61)', '62': '생명공학관(62)', '70': '대강당', '71': '의학관',
    '83': '제2종합연구동', '85': '산학협력센터', '86': 'N센터'
}

if not os.path.exists(MAP_PATH):
    print(f"Error: 지도 파일이 없습니다 ({MAP_PATH})")
    exit(1)

img = Image.open(MAP_PATH)
img_np = np.array(img)
H, W, _ = img_np.shape

fig, ax = plt.subplots(figsize=(16, 14))
ax.imshow(img_np)
ax.axis('off')

collected = {}

def on_click(event):
    if event.inaxes != ax or event.button != 1:
        return
    
    n = len(collected)
    if n >= len(CLICK_ORDER):
        return
    
    bld = CLICK_ORDER[n]
    x, y = int(event.xdata), int(event.ydata)
    # 계산용 좌표 (좌측 하단 0,0 기준)
    calc_y = H - y
    
    collected[bld] = (x, y, calc_y)
    
    # 지도에 표시
    ax.scatter(x, y, s=200, c='red', edgecolors='white', zorder=5)
    ax.text(x + 15, y, f"{bld}: {NAMES[bld]}", fontsize=9, color='red', fontweight='bold',
            bbox=dict(fc='white', alpha=0.8, ec='none', pad=1))
    
    remaining = len(CLICK_ORDER) - len(collected)
    if remaining > 0:
        next_bld = CLICK_ORDER[len(collected)]
        ax.set_title(f'Click [{next_bld}] {NAMES[next_bld]} ({remaining} left)', fontsize=14)
    else:
        ax.set_title('✅ All Done! Check terminal for coordinates.', fontsize=15, color='green', fontweight='bold')
        print_results()
    
    fig.canvas.draw_idle()

def print_results():
    print("\n" + "="*50)
    print("📍 정제된 건물 좌표 (campus_graph.py 업데이트용)")
    print("="*50)
    print("BUILDINGS = {")
    for bld in CLICK_ORDER:
        x, y, _ = collected[bld]
        print(f"    '{bld}': {{'name': '{NAMES[bld]}', 'campus_x': {x}, 'campus_y': {y}}},")
    print("}")
    print("="*50)
    print("\n📏 거리 계산용 (좌측 하단 0,0 기준)")
    for bld in CLICK_ORDER:
        x, _, cy = collected[bld]
        print(f"'{bld}': ({x}, {cy})")

fig.canvas.mpl_connect('button_press_event', on_click)
ax.set_title(f'Click [03] {NAMES["03"]} ({len(CLICK_ORDER)} total)', fontsize=14)

print("시작하려면 창을 열고 지도를 클릭하세요.")
print(f"클릭 순서: {', '.join(CLICK_ORDER)}")
plt.show()
