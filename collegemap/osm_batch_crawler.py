# -*- coding: utf-8 -*-
"""
osm_batch_crawler.py  —  다중 캠퍼스 OSM 배치 수집기

이미 images/ 에 파일이 있으면 자동 스킵.
실행: python collegemap/osm_batch_crawler.py

수집 전략 (우선순위순):
  1. features_from_place      — OSM relation/way boundary 있는 경우
  2. amenity=university 폴리곤 — 캠퍼스 경계가 amenity 태그로 등록된 경우
  3. features_from_point 500m — boundary 없을 때 반경 fallback (300개 초과 시 스킵)
"""

import os
import time
import traceback
import osmnx as ox
from PIL import Image, ImageDraw

W, H   = 2223, 2056
MARGIN = 0.92

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
IMG_DIR  = os.path.join(BASE_DIR, "images")
TXT_DIR  = os.path.join(BASE_DIR, "txt")
os.makedirs(IMG_DIR, exist_ok=True)
os.makedirs(TXT_DIR, exist_ok=True)

# ── 수집할 캠퍼스 목록 (place, slug) ──────────────────────────────────────────
CAMPUSES = [
    # ── 한국: 기존분 (이미 있으면 자동 스킵) ──────────────────────────────
    ("Seoul National University, Seoul, South Korea",           "seoul_national_university"),
    ("Kwangwoon University, Seoul, South Korea",                "kwangwoon_university"),
    ("Hankuk University of Foreign Studies, Seoul, South Korea","hankuk_university_of_foreign_studies"),
    ("Duksung Women's University, Seoul, South Korea",          "duksung_womens_university"),
    ("Dongduk Women's University, Seoul, South Korea",          "dongduk_womens_university"),
    ("Seoul Women's University, Seoul, South Korea",            "seoul_womens_university"),
    ("Sahmyook University, Seoul, South Korea",                 "sahmyook_university"),
    ("Hansung University, Seoul, South Korea",                  "hansung_university"),
    ("Myongji University, Seoul, South Korea",                  "myongji_university"),
    ("Sangmyung University, Seoul, South Korea",                "sangmyung_university"),
    ("Korea Aerospace University, Goyang, South Korea",         "korea_aerospace_university"),
    ("Ajou University, Suwon, South Korea",                     "ajou_university"),
    ("Inha University, Incheon, South Korea",                   "inha_university"),
    ("Incheon National University, Incheon, South Korea",       "incheon_national_university"),
    ("Gachon University, Seongnam, South Korea",                "gachon_university"),
    ("Dankook University, Yongin, South Korea",                 "dankook_university"),
    ("KAIST, Daejeon, South Korea",                             "kaist"),
    ("POSTECH, Pohang, South Korea",                            "postech"),
    ("Chungnam National University, Daejeon, South Korea",      "chungnam_national_university"),
    ("Pusan National University, Busan, South Korea",           "pusan_national_university"),
    ("Kyungpook National University, Daegu, South Korea",       "kyungpook_national_university"),
    ("Chonnam National University, Gwangju, South Korea",       "chonnam_national_university"),
    ("Jeju National University, Jeju, South Korea",             "jeju_national_university"),
    ("Kyung Hee University, Yongin, South Korea",               "kyung_hee_university_international"),
    ("Hanyang University, Ansan, South Korea",                  "hanyang_university_erica"),

    # ── 한국: 신규 거점 국립대 ─────────────────────────────────────────────
    ("Kangwon National University, Chuncheon, South Korea",     "kangwon_national_university"),
    ("Jeonbuk National University, Jeonju, South Korea",        "jeonbuk_national_university"),
    ("Chungbuk National University, Cheongju, South Korea",     "chungbuk_national_university"),
    ("Gyeongsang National University, Jinju, South Korea",      "gyeongsang_national_university"),
    ("Pukyong National University, Busan, South Korea",         "pukyong_national_university"),
    ("Hanbat National University, Daejeon, South Korea",        "hanbat_national_university"),

    # ── 한국: 신규 2캠퍼스 / 지방 사립 ───────────────────────────────────
    ("Yonsei University, Incheon, South Korea",                 "yonsei_university_international"),
    ("Korea University, Sejong, South Korea",                   "korea_university_sejong"),
    ("Hankuk University of Foreign Studies, Yongin, South Korea","hufs_global_campus"),
    ("Chosun University, Gwangju, South Korea",                 "chosun_university"),
    ("Dong-A University, Busan, South Korea",                   "dong_a_university"),
    ("Yeungnam University, Gyeongsan, South Korea",             "yeungnam_university"),
    ("Hallym University, Chuncheon, South Korea",               "hallym_university"),
    ("Keimyung University, Daegu, South Korea",                 "keimyung_university"),
    ("Inje University, Gimhae, South Korea",                    "inje_university"),

    # ── 일본 (OSM 품질 최고) ───────────────────────────────────────────────
    ("University of Tokyo, Bunkyo, Tokyo, Japan",               "university_of_tokyo"),
    ("Kyoto University, Kyoto, Japan",                          "kyoto_university"),
    ("Osaka University, Suita, Osaka, Japan",                   "osaka_university"),
    ("Waseda University, Shinjuku, Tokyo, Japan",               "waseda_university"),
    ("Keio University, Minato, Tokyo, Japan",                   "keio_university"),
    ("Tokyo Institute of Technology, Meguro, Tokyo, Japan",     "tokyo_institute_of_technology"),
    ("Tohoku University, Sendai, Japan",                        "tohoku_university"),
    ("Nagoya University, Nagoya, Japan",                        "nagoya_university"),
    ("Hokkaido University, Sapporo, Japan",                     "hokkaido_university"),
    ("Kyushu University, Fukuoka, Japan",                       "kyushu_university"),
    ("Kobe University, Kobe, Japan",                            "kobe_university"),
    ("Hiroshima University, Higashihiroshima, Japan",           "hiroshima_university"),

    # ── 미국 ──────────────────────────────────────────────────────────────
    ("Massachusetts Institute of Technology, Cambridge, Massachusetts, United States", "mit"),
    ("Stanford University, Stanford, California, United States","stanford_university"),
    ("Harvard University, Cambridge, Massachusetts, United States","harvard_university"),
    ("University of California Berkeley, Berkeley, California, United States","uc_berkeley"),
    ("University of Michigan, Ann Arbor, Michigan, United States","university_of_michigan"),
    ("Georgia Institute of Technology, Atlanta, Georgia, United States","georgia_tech"),
    ("Cornell University, Ithaca, New York, United States",     "cornell_university"),
    ("Carnegie Mellon University, Pittsburgh, Pennsylvania, United States","carnegie_mellon_university"),
    ("University of Illinois Urbana-Champaign, Champaign, Illinois, United States","uiuc"),
    ("Princeton University, Princeton, New Jersey, United States","princeton_university"),

    # ── 유럽 ──────────────────────────────────────────────────────────────
    ("ETH Zurich, Zurich, Switzerland",                         "eth_zurich"),
    ("TU Delft, Delft, Netherlands",                            "tu_delft"),
    ("Imperial College London, London, United Kingdom",         "imperial_college_london"),
    ("Technische Universität München, Munich, Germany",         "tu_munich"),
    ("EPFL, Lausanne, Switzerland",                             "epfl"),
    ("University of Edinburgh, Edinburgh, United Kingdom",      "university_of_edinburgh"),
    ("KU Leuven, Leuven, Belgium",                              "ku_leuven"),
    ("Uppsala University, Uppsala, Sweden",                     "uppsala_university"),
    ("Ghent University, Ghent, Belgium",                        "ghent_university"),
    ("University of Amsterdam, Amsterdam, Netherlands",         "university_of_amsterdam"),

    # ── 아시아·오세아니아·캐나다 ──────────────────────────────────────────
    ("National University of Singapore, Singapore",             "national_university_of_singapore"),
    ("Tsinghua University, Beijing, China",                     "tsinghua_university"),
    ("Peking University, Beijing, China",                       "peking_university"),
    ("University of Melbourne, Melbourne, Australia",           "university_of_melbourne"),
    ("University of Sydney, Sydney, Australia",                 "university_of_sydney"),
    ("University of Toronto, Toronto, Canada",                  "university_of_toronto"),
    ("McGill University, Montreal, Canada",                     "mcgill_university"),
    ("University of British Columbia, Vancouver, Canada",       "university_of_british_columbia"),
]


# ── 수집 로직 ─────────────────────────────────────────────────────────────────
POINT_FALLBACK_DIST = 500
MAX_BUILDINGS_POINT = 300


def _fetch_buildings(place: str):
    """
    1순위: features_from_place (OSM boundary relation)
    2순위: amenity=university 폴리곤 내 건물 쿼리
    3순위: features_from_point 반경 500m fallback
    """
    # 1순위
    try:
        gdf = ox.features_from_place(place, tags={"building": True})
        gdf = gdf[gdf.geometry.geom_type.isin(["Polygon", "MultiPolygon"])]
        if not gdf.empty:
            return gdf, "place"
    except Exception:
        pass

    # 2순위: geocode → amenity=university 폴리곤 찾기 → 그 안의 건물 쿼리
    try:
        loc = ox.geocode(place)
        campus_gdf = ox.features_from_point(loc, dist=2000, tags={"amenity": "university"})
        campus_polys = campus_gdf[
            campus_gdf.geometry.geom_type.isin(["Polygon", "MultiPolygon"])
        ]
        if not campus_polys.empty:
            # 중심에서 가장 가까운 폴리곤 선택
            from shapely.geometry import Point
            center = Point(loc[1], loc[0])
            campus_polys = campus_polys.copy()
            campus_polys["_dist"] = campus_polys.geometry.centroid.distance(center)
            poly = campus_polys.sort_values("_dist").geometry.iloc[0]
            gdf = ox.features_from_polygon(poly, tags={"building": True})
            gdf = gdf[gdf.geometry.geom_type.isin(["Polygon", "MultiPolygon"])]
            if not gdf.empty:
                return gdf, "polygon"
    except Exception:
        pass

    # 3순위: 반경 fallback
    try:
        loc = ox.geocode(place)
        gdf = ox.features_from_point(loc, dist=POINT_FALLBACK_DIST, tags={"building": True})
        gdf = gdf[gdf.geometry.geom_type.isin(["Polygon", "MultiPolygon"])]
        return gdf, "point"
    except Exception:
        return None, "fail"


def process(place: str, slug: str) -> bool:
    img_path = os.path.join(IMG_DIR, f"{slug}_building_mask.png")
    txt_path = os.path.join(TXT_DIR, f"{slug}_building_places.txt")

    if os.path.exists(img_path):
        print(f"  [SKIP] 이미 있음: {slug}")
        return True

    print(f"  ▶ 수집: {place}")
    gdf, method = _fetch_buildings(place)

    if gdf is None or gdf.empty:
        print(f"  [FAIL] 건물 없음")
        return False

    if method == "point":
        print(f"  [INFO] fallback 반경 {POINT_FALLBACK_DIST}m ({len(gdf)} 건물)")
        if len(gdf) > MAX_BUILDINGS_POINT:
            print(f"  [SKIP] {len(gdf)}개 → 주변 포함 의심, 스킵")
            return False
    else:
        print(f"  [INFO] method={method}, {len(gdf)} 건물")

    gdf_proj = gdf.to_crs(gdf.estimate_utm_crs())
    minx, miny, maxx, maxy = gdf_proj.total_bounds
    scale = min(W / (maxx - minx), H / (maxy - miny)) * MARGIN
    offx  = (W - (maxx - minx) * scale) / 2
    offy  = (H - (maxy - miny) * scale) / 2

    BUILDING_POLY = {}
    for i, (_, row) in enumerate(gdf_proj.iterrows()):
        geom = row.geometry
        if geom.geom_type == "MultiPolygon":
            geom = max(geom.geoms, key=lambda g: g.area)
        pts = [(int((x - minx) * scale + offx), int((maxy - y) * scale + offy))
               for x, y in geom.exterior.coords]
        if len(pts) > 1 and pts[0] == pts[-1]:
            pts.pop()
        BUILDING_POLY[str(i).zfill(2)] = pts

    img  = Image.new('L', (W, H), 0)
    draw = ImageDraw.Draw(img)
    for pts in BUILDING_POLY.values():
        draw.polygon(pts, fill=255)
    img.save(img_path)

    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("BUILDING_POLY = {\n")
        for k, pts in BUILDING_POLY.items():
            f.write(f"    '{k}': {pts},\n")
        f.write("}\n")

    print(f"  ✓ {slug}  ({len(BUILDING_POLY)} 건물)")
    return True


if __name__ == "__main__":
    total = len(CAMPUSES)
    ok, fail = 0, []

    print(f"▶ 총 {total}개 캠퍼스 수집 시작 (이미 있으면 자동 스킵)\n")
    for i, (place, slug) in enumerate(CAMPUSES, 1):
        print(f"[{i:2d}/{total}] {slug}")
        try:
            if process(place, slug):
                ok += 1
            else:
                fail.append(slug)
        except Exception as e:
            print(f"  [ERROR] {e}")
            traceback.print_exc()
            fail.append(slug)
        time.sleep(1)

    print(f"\n완료: 성공 {ok}개 / 실패 {len(fail)}개")
    if fail:
        print("실패:", fail)
