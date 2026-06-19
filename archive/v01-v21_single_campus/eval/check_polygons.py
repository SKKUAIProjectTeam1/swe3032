import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from PIL import Image
import numpy as np
import sys
import os

# 현재 경로 추가
sys.path.append(os.getcwd())

try:
    from campus_overlay import BUILDING_POLY, MAP_PATH
except ImportError:
    print("Error: campus_overlay.py 또는 필요한 변수를 찾을 수 없습니다.")
    sys.exit(1)

def main():
    if not os.path.exists(MAP_PATH):
        print(f"Error: 지도 파일이 없습니다 ({MAP_PATH})")
        return

    img = np.array(Image.open(MAP_PATH))
    fig, ax = plt.subplots(figsize=(15, 14))
    ax.imshow(img)

    for bld, pts in BUILDING_POLY.items():
        # 폴리곤 그리기 (반투명 사이언 색상)
        poly = mpatches.Polygon(pts, closed=True, facecolor='cyan', edgecolor='blue', alpha=0.4, linewidth=2)
        ax.add_patch(poly)
        
        # 건물 번호 표시
        pts_array = np.array(pts)
        cx, cy = np.mean(pts_array, axis=0)
        ax.text(cx, cy, bld, color='white', fontweight='bold', fontsize=12, 
                ha='center', va='center', bbox=dict(boxstyle='round,pad=0.2', fc='black', alpha=0.7))

    plt.title("Campus Building Polygon Verification", fontsize=20, pad=20)
    plt.axis('off')
    
    out_path = 'campus_polygons_check.png'
    plt.savefig(out_path, dpi=120, bbox_inches='tight')
    plt.close()
    print(f"✅ 시각화 완료: {out_path} 파일을 확인해 주세요.")

if __name__ == '__main__':
    main()
