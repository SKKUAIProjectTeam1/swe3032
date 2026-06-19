# Neural Civil Engineer — GNN 기반 캠퍼스 도로망 자율 설계

> **"GNN에게 건물 배치(기하학) 정보만 주면, 합리적인 도로망을 스스로 설계할 수 있는가?"**

건물 폴리곤 이미지만 입력으로 받아, 모든 건물군과 게이트를 연결하는 도로망을 자동으로 예측하는 GNN 모델.  
사람의 이동 데이터나 시간표 없이, **건물 배치의 기하학적 구조만**으로 연결성 있는 도로망 후보를 생성한다.

---

## 최종 결과 요약

### 6-모델 비교 (Val 7 + Test 37 = 44개 캠퍼스 기준)

| 모델 | 설명 | Recall | F1 | Connectivity Rate | Edge Density | Collision |
|------|------|--------|-----|------------------|--------------|-----------|
| Trivial (전체 예측) | 모든 free 엣지를 도로로 예측 (하한선) | 1.000 | ≈0.019 | 100.0% | 100.0% | 0% |
| MST | 게이트+건물군을 최소 비용으로 연결 | 0.037 | 0.044 | 99.4% | 0.5% | 0% |
| Ridge-only (thr=0.1) | ridge 점수 기반 규칙 예측 | 0.086 | 0.046 | 94.4% | 2.6% | 0% |
| SAGE | Mean aggregation GNN | 0.714 | ≈0.106 | 97.3% | 18.3% | 0% |
| MLP | 이웃 정보 미사용 MLP | 0.713 | ≈0.100 | 98.7% | 19.9% | 0% |
| **V28a (최종, GAT)** | **Subgraph GNN + 16-vote GT** | **0.724** | **≈0.104** | **97.9%** | **18.0%** | **0%** |

> Recall/F1은 Val 7개(GT 있음) 기준. Connectivity Rate/Edge Density는 Val+Test 44개 전체 기준.

### Precision 비교 (Val 7개 기준)

| 모델 | Precision |
|------|-----------|
| Trivial | ≈0.010 |
| MST | 0.073 |
| Ridge-only | 0.037 |
| MLP | 0.054 |
| V28a (GAT) | 0.056 |
| SAGE | 0.057 |

---

## 데이터셋

- **규모**: 국내·해외 49개 대학 캠퍼스 건물 배치 (이미지 + 건물 폴리곤 좌표)
- **입력 형태**: 100×100 격자, 건물이 아닌 free pixel만 노드로 사용 (캠퍼스당 평균 ~9,000 노드, ~40,000 엣지)
- **Train/Val/Test**: 168개(증강 포함) / 7개 / 37개(GT 없음)
  - GT 있는 42개 원본 캠퍼스 → 4방향 flip 증강(lr/ud/both) → 168개 Train
  - GT 없는 37개 캠퍼스 → Test (구조적 지표만 평가)

### GT(정답) 생성 — 16-vote 파이프라인

단순히 "정답 도로 지도"가 있는 게 아니라, **여러 알고리즘의 합의 결과**를 정답으로 사용한다.

```text
4개 진입 방향(게이트) × 4가지 비용함수(Dijkstra 3종 + A*) = 16가지 경로 후보
  ↓
합의 기준(c4/c8/c12: 4·8·12개 이상 동의)으로 3가지 마스크 생성
  ↓
캠퍼스별로 사람이 가장 적절한 마스크 선택 → gt_masks_final/
```

데이터 파일: `collegemap/gt_masks_final/*_gt.npz`

---

## 최종 모델 — V28a Subgraph GAT

### 핵심 변화 (V28 이전 대비)

| 항목 | 이전 (V22~V27) | V28a |
|------|--------------|------|
| 그래프 노드 | 10,000개 (건물 포함) | ~9,000개 (free pixel만) |
| GT edge 비율 | 0.025% | ~0.5% (약 20배 개선) |
| 클러스터링 | single-linkage | ball clustering (체인 방지) |
| 증강 | 없음 | 4방향 flip |
| 손실 함수 | BCE+Dice+FP+Sparse | + Connectivity Loss |

### 모델 구조

- **3-layer Multi-Head GAT** (64차원, 4 heads)
- **엣지 분류기**: `[h_src, h_dst, |h_src-h_dst|, h_src*h_dst, ridge(2)]` → MLP → sigmoid
- **파라미터 수**: ~47k
- **노드 피처** (9차원): ridge, 건물까지 거리, x/y 좌표, cluster_indicator, dist_to_cluster, dist_to_gate, 방향벡터(dy/dx)

### 손실 함수 (5개 항목의 합)

| 항목 | 역할 | 가중치 |
|------|------|--------|
| `L_BCE` | 도로/비도로 분류. pos_weight 최대 80배(GT 비율 기반 자동 계산) | 1.0 |
| `L_Dice` | 예측-정답 겹침(overlap)을 직접 최적화 | 0.8 |
| `L_FP` | 건물 근처(ridge 낮은 곳) 오탐 억제. warm-up: 0.2→1.0 (20 epoch) | warm-up |
| `L_conn` | 모든 터미널(게이트+건물군)이 도로망에 연결되도록 강제. warm-up: 0.05→0.5 | warm-up |
| `L_sparse` | 과도한 도로 예측 억제 | 0.03 |

```text
L = L_BCE + 0.8·L_Dice + (0.2+0.8s)·L_FP + (0.05+0.45s)·L_conn + 0.03·L_sparse
    (s: warm-up 진행도 0→1, 20 epoch 동안)
```

### 학습 설정

| 항목 | 값 |
|------|-----|
| Optimizer | AdamW |
| LR Schedule | Cosine Annealing |
| Epochs | 220 |
| Learning Rate | 4e-4 |
| Warm-up | 20 epoch |

---

## 실행 방법

> **학습/추론은 Google Colab(GPU)에서 실행 권장.**  
> 로컬 환경에서는 평가 스크립트만 실행 가능.

### 학습 (Colab)

세 가지 모델 각각의 노트북을 Colab에서 열어 실행:

| 파일 | 설명 |
|------|------|
| `train_v28_subgraph_gnn.ipynb` | **최종 모델 (GAT)** |
| `train_v28_subgraph_mlp.ipynb` | MLP 비교군 |
| `train_v28_subgraph_sage.ipynb` | SAGE 비교군 |

각 노트북 상단에서 Google Drive 연동 후 실행. 학습 완료 시 체크포인트가 `v28_final/checkpoints/` 하위에 저장됨.

### 평가 — 6-모델 비교 (로컬)

```bash
# swe3032/ 디렉토리에서 실행
python scripts/connectivity_eval.py
```

Connectivity Rate + Edge Density를 6가지 모델 전체(Trivial/MST/Ridge-only/SAGE/MLP/V28a)에 대해 일괄 계산. GT가 없는 Test 캠퍼스도 평가에 포함됨.

### GT 사전 계산

```bash
python scripts/precompute_gt.py
```

---

## 파일 구조

```text
swe3032/
├── train_v28_subgraph_gnn.ipynb    ← 최종 모델(GAT) 학습
├── train_v28_subgraph_mlp.ipynb    ← MLP 비교군 학습
├── train_v28_subgraph_sage.ipynb   ← SAGE 비교군 학습
├── save_to_drive.ipynb
├── README.md / CONTEXT.md
│
├── scripts/                        ← 평가·전처리 스크립트
│   ├── connectivity_eval.py        # 6-모델 Connectivity/Density 평가
│   ├── baseline_eval.py            # Trivial/MST/Ridge-only 평가
│   ├── precompute_gt.py            # 16-vote GT 마스크 사전 계산
│   └── precompute_gt_*.py          # GT 변형 실험용
│
├── v28_final/                      ← 최종 모델 결과물
│   ├── checkpoints/
│   │   ├── gat/                    # GAT 체크포인트 (best + ep050~ep220)
│   │   ├── mlp/                    # MLP 체크포인트
│   │   └── sage/                   # SAGE 체크포인트
│   └── results/
│       ├── gat_v28/                # V28 GAT 예측 결과 이미지
│       ├── gat_v28a/               # V28a GAT 예측 결과 이미지
│       ├── mlp/                    # MLP 예측 결과 이미지
│       └── sage/                   # SAGE 예측 결과 이미지
│
├── collegemap/                     ← 데이터셋
│   ├── images/                     # 캠퍼스 건물 마스크 이미지 (49개)
│   ├── txt/                        # 건물 폴리곤 좌표 (49개)
│   ├── gt_masks_final/             # 최종 GT 마스크 (*_gt.npz, 42개)
│   └── osm_campus_converter.py     # OSM → 건물 폴리곤 변환
│
├── docs/                           ← 개발 기록 문서
│
├── archive/
│   ├── v01-v21_single_campus/      ← 단일 캠퍼스 직접 최적화 시대
│   └── v22-v27_multicampus/        ← 멀티캠퍼스 GAT 실험 시대
│
└── legacy/oldversion/              ← 이전 프로젝트 (혼잡도 예측)
```

---

## 개선 과정 요약

| 버전대 | 시도 | 핵심 문제 |
|--------|------|-----------|
| V1~V21 | 단일 캠퍼스 직접 최적화 | 분절된 도로 구간, 다른 캠퍼스로 일반화 불가 |
| V22~V27 | 멀티캠퍼스 GAT + ridge GT | "지렁이" 형태 우회 도로, GT edge 비율 0.025%로 학습 정체 |
| **V28a** | **non-building subgraph + 16-vote GT** | GT edge 비율 0.5%로 20배 개선, 연결성 보장 |

---

## 팀 정보

성균관대학교 소프트웨어학과 | 인공지능프로젝트 1조  
GitHub: [SKKUAIProjectTeam1/swe3032](https://github.com/SKKUAIProjectTeam1/swe3032)
