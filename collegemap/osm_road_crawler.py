# -*- coding: utf-8 -*-
"""
osm_road_crawler.py — OSM에서 도로 GT 마스크 수집

건물 마스크와 동일한 UTM 변환을 사용해 도로를 100x100 그리드에 정확히 정렬합니다.
출력: collegemap/road_masks/{slug}_road_mask.npy  (100x100, float32, 0~1)

실행: python collegemap/osm_road_crawler.py
"""

import os
import time
import traceback
import numpy as np
import osmnx as ox
from PIL import Image, ImageDraw

W, H   = 2223, 2056
MARGIN = 0.92
RES    = 100

BASE_DIR  = os.path.dirname(os.path.abspath(__file__))
IMG_DIR   = os.path.join(BASE_DIR, "images")
ROAD_DIR  = os.path.join(BASE_DIR, "road_masks")
os.makedirs(ROAD_DIR, exist_ok=True)

# 가장 중요한 도로 = tier 3, 중간 = tier 2, 보행로 = tier 1
# tier 0은 무시 (steps, track, proposed 등)
ROAD_TIERS = {
    'primary': 3, 'primary_link': 3,
    'secondary': 3, 'secondary_link': 3,
    'tertiary': 2, 'tertiary_link': 2,
    'residential': 2, 'living_street': 2,
    'service': 1,
    'pedestrian': 2, 'footway': 1, 'path': 1, 'cycleway': 1,
    'unclassified': 1,
}

# ── 크롤러에서 가져온 place→slug 매핑 ─────────────────────────────────────────
CAMPUSES = [
    ("Seoul National University, Seoul, South Korea",            "seoul_national_university"),
    ("Kwangwoon University, Seoul, South Korea",                 "kwangwoon_university"),
    ("Hankuk University of Foreign Studies, Seoul, South Korea", "hankuk_university_of_foreign_studies"),
    ("Duksung Women's University, Seoul, South Korea",           "duksung_womens_university"),
    ("Dongduk Women's University, Seoul, South Korea",           "dongduk_womens_university"),
    ("Seoul Women's University, Seoul, South Korea",             "seoul_womens_university"),
    ("Sahmyook University, Seoul, South Korea",                  "sahmyook_university"),
    ("Hansung University, Seoul, South Korea",                   "hansung_university"),
    ("Myongji University, Seoul, South Korea",                   "myongji_university"),
    ("Sangmyung University, Seoul, South Korea",                 "sangmyung_university"),
    ("Korea Aerospace University, Goyang, South Korea",          "korea_aerospace_university"),
    ("Ajou University, Suwon, South Korea",                      "ajou_university"),
    ("Inha University, Incheon, South Korea",                    "inha_university"),
    ("Incheon National University, Incheon, South Korea",        "incheon_national_university"),
    ("Gachon University, Seongnam, South Korea",                 "gachon_university"),
    ("Dankook University, Yongin, South Korea",                  "dankook_university"),
    ("KAIST, Daejeon, South Korea",                              "kaist"),
    ("POSTECH, Pohang, South Korea",                             "postech"),
    ("Chungnam National University, Daejeon, South Korea",       "chungnam_national_university"),
    ("Pusan National University, Busan, South Korea",            "pusan_national_university"),
    ("Kyungpook National University, Daegu, South Korea",        "kyungpook_national_university"),
    ("Chonnam National University, Gwangju, South Korea",        "chonnam_national_university"),
    ("Jeju National University, Jeju, South Korea",              "jeju_national_university"),
    ("Kyung Hee University, Yongin, South Korea",                "kyung_hee_university_international"),
    ("Hanyang University, Ansan, South Korea",                   "hanyang_university_erica"),
    ("Kangwon National University, Chuncheon, South Korea",      "kangwon_national_university"),
    ("Jeonbuk National University, Jeonju, South Korea",         "jeonbuk_national_university"),
    ("Chungbuk National University, Cheongju, South Korea",      "chungbuk_national_university"),
    ("Gyeongsang National University, Jinju, South Korea",       "gyeongsang_national_university"),
    ("Pukyong National University, Busan, South Korea",          "pukyong_national_university"),
    ("Hanbat National University, Daejeon, South Korea",         "hanbat_national_university"),
    ("Yonsei University, Incheon, South Korea",                  "yonsei_university_international"),
    ("Korea University, Sejong, South Korea",                    "korea_university_sejong"),
    ("Hankuk University of Foreign Studies, Yongin, South Korea","hufs_global_campus"),
    ("Chosun University, Gwangju, South Korea",                  "chosun_university"),
    ("Dong-A University, Busan, South Korea",                    "dong_a_university"),
    ("Yeungnam University, Gyeongsan, South Korea",              "yeungnam_university"),
    ("Hallym University, Chuncheon, South Korea",                "hallym_university"),
    ("Keimyung University, Daegu, South Korea",                  "keimyung_university"),
    ("Inje University, Gimhae, South Korea",                     "inje_university"),
    ("University of Tokyo, Bunkyo, Tokyo, Japan",                "university_of_tokyo"),
    ("Kyoto University, Kyoto, Japan",                           "kyoto_university"),
    ("Osaka University, Suita, Osaka, Japan",                    "osaka_university"),
    ("Waseda University, Shinjuku, Tokyo, Japan",                "waseda_university"),
    ("Keio University, Minato, Tokyo, Japan",                    "keio_university"),
    ("Tokyo Institute of Technology, Meguro, Tokyo, Japan",      "tokyo_institute_of_technology"),
    ("Tohoku University, Sendai, Japan",                         "tohoku_university"),
    ("Nagoya University, Nagoya, Japan",                         "nagoya_university"),
    ("Hokkaido University, Sapporo, Japan",                      "hokkaido_university"),
    ("Kyushu University, Fukuoka, Japan",                        "kyushu_university"),
    ("Kobe University, Kobe, Japan",                             "kobe_university"),
    ("Hiroshima University, Higashihiroshima, Japan",            "hiroshima_university"),
    ("Massachusetts Institute of Technology, Cambridge, Massachusetts, United States", "mit"),
    ("Stanford University, Stanford, California, United States", "stanford_university"),
    ("Harvard University, Cambridge, Massachusetts, United States","harvard_university"),
    ("University of California Berkeley, Berkeley, California, United States","uc_berkeley"),
    ("University of Michigan, Ann Arbor, Michigan, United States","university_of_michigan"),
    ("Georgia Institute of Technology, Atlanta, Georgia, United States","georgia_tech"),
    ("Carnegie Mellon University, Pittsburgh, Pennsylvania, United States","carnegie_mellon_university"),
    ("University of Illinois Urbana-Champaign, Champaign, Illinois, United States","uiuc"),
    ("Princeton University, Princeton, New Jersey, United States","princeton_university"),
    ("ETH Zurich, Zurich, Switzerland",                          "eth_zurich"),
    ("TU Delft, Delft, Netherlands",                             "tu_delft"),
    ("Imperial College London, London, United Kingdom",          "imperial_college_london"),
    ("Technische Universität München, Munich, Germany",          "tu_munich"),
    ("EPFL, Lausanne, Switzerland",                              "epfl"),
    ("University of Edinburgh, Edinburgh, United Kingdom",       "university_of_edinburgh"),
    ("KU Leuven, Leuven, Belgium",                               "ku_leuven"),
    ("Uppsala University, Uppsala, Sweden",                      "uppsala_university"),
    ("Ghent University, Ghent, Belgium",                         "ghent_university"),
    ("University of Amsterdam, Amsterdam, Netherlands",          "university_of_amsterdam"),
    ("National University of Singapore, Singapore",              "national_university_of_singapore"),
    ("Tsinghua University, Beijing, China",                      "tsinghua_university"),
    ("Peking University, Beijing, China",                        "peking_university"),
    ("University of Melbourne, Melbourne, Australia",            "university_of_melbourne"),
    ("University of Sydney, Sydney, Australia",                  "university_of_sydney"),
    ("University of Toronto, Toronto, Canada",                   "university_of_toronto"),
    ("McGill University, Montreal, Canada",                      "mcgill_university"),
    ("University of British Columbia, Vancouver, Canada",        "university_of_british_columbia"),
    # 수동 생성 캠퍼스
    ("Aalto University, Espoo, Finland",                         "aalto_university"),
    ("Sungkyunkwan University, Seoul, South Korea",              "sungkyunkwan_university"),
    ("Sungkyunkwan University, Suwon, South Korea",              "sungkyunkwan_university_natural_science"),
    ("Korea University, Seoul, South Korea",                     "korea_university"),
    ("Yonsei University, Seoul, South Korea",                    "yonsei_university"),
    ("Hanyang University, Seoul, South Korea",                   "hanyang_university"),
    ("Kyung Hee University, Seoul, South Korea",                 "kyung_hee_university"),
    ("Ewha Womans University, Seoul, South Korea",               "ewha_womans_university"),
    ("Hongik University, Seoul, South Korea",                    "hongik_university"),
    ("Sogang University, Seoul, South Korea",                    "sogang_university"),
    ("Konkuk University, Seoul, South Korea",                    "konkuk_university"),
    ("Dongguk University, Seoul, South Korea",                   "dongguk_university"),
    ("Kookmin University, Seoul, South Korea",                   "kookmin_university"),
    ("Sejong University, Seoul, South Korea",                    "sejong_university"),
    ("Soongsil University, Seoul, South Korea",                  "soongsil_university"),
    ("University of Groningen, Groningen, Netherlands",          "university_of_groningen"),
    ("University of Seoul, Seoul, South Korea",                  "university_of_seoul"),
]

# slug → place 역방향 맵
SLUG_TO_PLACE = {slug: place for place, slug in CAMPUSES}


def _get_utm_transform(place):
    """건물 데이터로 UTM 변환 파라미터 계산 (건물 마스크와 정렬 보장)."""
    gdf = ox.features_from_place(place, tags={"building": True})
    gdf = gdf[gdf.geometry.geom_type.isin(["Polygon", "MultiPolygon"])]
    if gdf.empty:
        raise ValueError(f"No buildings found: {place}")
    utm_crs = gdf.estimate_utm_crs()
    gdf_proj = gdf.to_crs(utm_crs)
    minx, miny, maxx, maxy = gdf_proj.total_bounds
    scale = min(W / (maxx - minx), H / (maxy - miny)) * MARGIN
    offx  = (W - (maxx - minx) * scale) / 2
    offy  = (H - (maxy - miny) * scale) / 2
    return utm_crs, minx, miny, maxx, maxy, scale, offx, offy


def _rasterize_roads(place, utm_crs, minx, miny, maxx, maxy):
    """OSM 도로를 RES×RES 그리드에 직접 래스터화.
    W×H로 그린 뒤 축소하면 가는 선이 사라지므로, 처음부터 100×100에 그림."""
    scale_r = min(RES / (maxx - minx), RES / (maxy - miny)) * MARGIN
    offx_r  = (RES - (maxx - minx) * scale_r) / 2
    offy_r  = (RES - (maxy - miny) * scale_r) / 2

    G = ox.graph_from_place(place, network_type='all', retain_all=True)
    edges = ox.graph_to_gdfs(G, nodes=False)
    edges_proj = edges.to_crs(utm_crs)

    road_img = Image.new('L', (RES, RES), 0)
    draw = ImageDraw.Draw(road_img)
    n_drawn = 0

    for _, row in edges_proj.iterrows():
        hw = row.get('highway', '')
        if isinstance(hw, list): hw = hw[0]
        tier = ROAD_TIERS.get(str(hw), 0)
        if tier == 0: continue

        geom = row.geometry
        if geom is None or geom.is_empty: continue
        coords = list(geom.coords) if hasattr(geom, 'coords') else []
        if len(coords) < 2: continue

        pts = []
        for x, y in coords:
            px = (x - minx) * scale_r + offx_r
            py = (maxy - y) * scale_r + offy_r
            pts.append((px, py))

        brightness = 85 + tier * 56  # tier1=141, tier2=197, tier3=253
        lw = tier + 1                 # tier1=2px, tier2=3px, tier3=4px (100×100 기준 충분)
        for i in range(len(pts) - 1):
            draw.line([pts[i], pts[i+1]], fill=brightness, width=lw)
        n_drawn += 1

    return road_img, n_drawn


def process(place, slug):
    out_path = os.path.join(ROAD_DIR, f"{slug}_road_mask.npy")
    if os.path.exists(out_path):
        print(f"  [SKIP] {slug}")
        return True

    img_path = os.path.join(IMG_DIR, f"{slug}_building_mask.png")
    if not os.path.exists(img_path):
        print(f"  [SKIP] 건물 마스크 없음: {slug}")
        return False

    print(f"  ▶ {slug} ({place})")
    try:
        utm_crs, minx, miny, maxx, maxy, scale, offx, offy = _get_utm_transform(place)
        road_img, n = _rasterize_roads(place, utm_crs, minx, miny, maxx, maxy)
        road_100 = np.array(road_img, dtype=np.float32) / 255.0
        np.save(out_path, road_100)
        print(f"  ✓ {slug}  (도로 {n}개, GT pixels={int((road_100 > 0.1).sum())})")
        return True
    except Exception as e:
        print(f"  [FAIL] {e}")
        return False


if __name__ == "__main__":
    import glob
    all_imgs = {os.path.basename(p).replace('_building_mask.png', '')
                for p in glob.glob(os.path.join(IMG_DIR, '*_building_mask.png'))}

    ok, fail, skip_no_place = 0, [], []
    total = len(all_imgs)
    print(f"▶ 총 {total}개 캠퍼스 도로 수집 시작\n")

    for slug in sorted(all_imgs):
        place = SLUG_TO_PLACE.get(slug)
        if place is None:
            skip_no_place.append(slug)
            print(f"  [NO_PLACE] {slug}")
            continue
        try:
            if process(place, slug):
                ok += 1
            else:
                fail.append(slug)
        except Exception as e:
            print(f"  [ERROR] {slug}: {e}")
            fail.append(slug)
        time.sleep(1)

    print(f"\n완료: 성공 {ok}개 / 실패 {len(fail)}개 / place없음 {len(skip_no_place)}개")
    if fail:         print("실패:", fail)
    if skip_no_place: print("place없음:", skip_no_place)
