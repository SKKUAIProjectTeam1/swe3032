# GNN 기반 캠퍼스 도로망 자동 설계

GNN(Graph Neural Network)을 활용해 대학 캠퍼스의 보행 경로 도로망을 자동으로 설계하는 시스템.
건물 폴리곤 데이터를 입력으로 받아, 건물 충돌을 피하고 모든 건물을 연결하는 최적의 도로망과 관문(Gate) 위치를 학습한다.

---

## 폴더 구조

```
swe3032/
├── building_places.txt       # 성균관대 자연과학캠퍼스 건물 폴리곤 좌표 (메인 입력)
├── roads.txt                 # 도로 참조 데이터
│
├── model/                    # GNN 모델 스크립트
│   ├── campus_grid_gnn_designer.py   ← 현재 메인 (V21)
│   ├── cam_gat.py                    # GAT 변형 모델
│   ├── cgr_gnn.py                    # CGR-GNN 변형
│   ├── gat.py / gat2.py              # GAT 실험 버전
│   ├── campus_network_designer.py    # 구버전 도로 설계 (참고용)
│   ├── campus_pure_gnn_designer.py   # 구버전 순수 GNN
│   └── campus_realistic_designer.py  # 구버전 리얼리스틱
│
├── collegemap/               # 다중 캠퍼스 OSM 데이터
│   ├── osm_campus_converter.py       # OpenStreetMap → BUILDING_POLY 변환 스크립트
│   ├── images/                       # 각 대학 건물 마스크 이미지 (20개)
│   └── txt/                          # 각 대학 BUILDING_POLY txt (20개)
│
├── eval/                     # 평가 및 분석 스크립트
│   ├── evaluate_road_design.py
│   ├── check_polygons.py
│   └── convert_coords.py
│
├── calibrate/                # 캠퍼스 맵 캘리브레이션 도구
│   ├── campus_calibrate_click.py     # 클릭으로 좌표 보정
│   ├── campus_calibrate_polygon.py   # 건물 폴리곤 보정
│   ├── campus_calibrate_roads.py     # 도로 보정
│   └── campus_calibrate_server.py    # 보정 서버
│
├── output/                   # 생성된 도로 설계 이미지 아카이브 (v4~v21)
├── maps/                     # 성균관대 캠퍼스 참조 지도
├── docs/                     # 프로젝트 문서
└── oldversion/               # 구 혼잡도 예측 프로젝트 (미사용)
```

---

## 메인 모델 실행

```bash
# swe3032/ 디렉토리에서 실행
python model/campus_grid_gnn_designer.py
```

- 입력: `building_places.txt` (CWD 기준)
- 출력: `campus_v21_smart_gate_design.png` (CWD 기준)
- 학습: 50,000 steps, Adam optimizer

### V21 모델 특징 (SmartQuadGateGNN)

| 구성 요소 | 내용 |
|-----------|------|
| 그래프 | 100×100 격자, 8방향 엣지 |
| 손실함수 | Collision + Connectivity + Gate Diversity + Proximity + Ridge + Sparsity + Degree |
| 관문(Gate) | 4개, Greedy Selection으로 중첩 방지 |
| 관문 선정 방식 | repulsion 패널티 + 건물군 proximity reward |

---

## 다중 캠퍼스 데이터 추가 (collegemap)

```bash
# osm_campus_converter.py 내 place 변수를 변경 후 실행
python collegemap/osm_campus_converter.py
```

`place = "학교명, 도시, South Korea"` 한 줄만 바꾸면 다른 대학의 건물 폴리곤 데이터를 자동 수집한다.
출력 파일은 `collegemap/txt/<학교명>_building_places.txt` 포맷으로 저장되며, `building_places.txt`와 동일한 `BUILDING_POLY` 딕셔너리 형식이다.

현재 수집된 대학 (20개): 고려대, 연세대, 성균관대, 한양대, 중앙대, 이화여대, 서강대, 홍익대, 건국대, 동국대, 국민대, 세종대, 숙명여대, 숭실대, 성신여대, 서울과기대, 서울교대, 한국체대, 경희대, 서울시립대

---

## 모델 버전 히스토리 (output/ 참고)

| 버전 | 특징 |
|------|------|
| V4~V9 | 초기 실험 (중력, 조기종료, 확산 등) |
| V11~V15 | 구조 개선 (backbone, perimeter, multiscale) |
| V16 | Civil Engineer — Ridge 정렬, 2관문 |
| V17~V20 | GAT 실험 (triple gate, internal, quad gate) |
| **V21** | **현재 최신** — Smart Quad Gate, 4관문 중첩 방지 |
