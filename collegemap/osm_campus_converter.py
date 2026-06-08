# -*- coding: utf-8 -*-
import os
import osmnx as ox
from PIL import Image, ImageDraw

W, H = 2223, 2056
MARGIN = 0.92
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
IMG_DIR = os.path.join(BASE_DIR, "images")
TXT_DIR = os.path.join(BASE_DIR, "txt")

place = "Korea University, Seoul, South Korea"
gdf = ox.features_from_place(place, tags={"building": True})
gdf = gdf[gdf.geometry.geom_type.isin(["Polygon", "MultiPolygon"])]

if gdf.empty:
    raise ValueError(f"No building polygons found for place={place!r}")

# WGS84(위경도) -> 캠퍼스 위치에 맞는 UTM으로 자동 투영
gdf_proj = ox.project_gdf(gdf)
minx, miny, maxx, maxy = gdf_proj.total_bounds

scale = min(W / (maxx - minx), H / (maxy - miny)) * MARGIN
offx = (W - (maxx - minx) * scale) / 2
offy = (H - (maxy - miny) * scale) / 2

BUILDING_POLY = {}
for i, (idx, row) in enumerate(gdf_proj.iterrows()):
    geom = row.geometry
    if geom.geom_type == "MultiPolygon":
        geom = max(geom.geoms, key=lambda g: g.area)
    pts = []
    for (x, y) in geom.exterior.coords:
        px = (x - minx) * scale + offx
        py = (maxy - y) * scale + offy  # 이미지 좌표는 y가 아래로 증가하므로 반전
        pts.append((int(px), int(py)))
    BUILDING_POLY[str(i).zfill(2)] = pts

print(f"변환된 건물 폴리곤 개수: {len(BUILDING_POLY)}")
print("샘플:", list(BUILDING_POLY.items())[0])

img = Image.new('L', (W, H), 0)
draw = ImageDraw.Draw(img)
for pts in BUILDING_POLY.values():
    draw.polygon(pts, fill=255)
out_path = os.path.join(IMG_DIR, "korea_univ_building_mask.png")
img.save(out_path)
print("저장:", out_path)

with open(os.path.join(TXT_DIR, "korea_univ_building_places.txt"), "w", encoding="utf-8") as f:
    f.write("BUILDING_POLY = {\n")
    for k, pts in BUILDING_POLY.items():
        f.write(f"    '{k}': {pts},\n")
    f.write("}\n")
print("저장: korea_univ_building_places.txt (기존 building_places.txt와 동일 포맷)")
