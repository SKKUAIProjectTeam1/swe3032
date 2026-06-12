# SWE3032 프로젝트 컨텍스트

## 목표
캠퍼스 건물 배치 이미지 + 건물 위치 정보를 입력받아,
**캠퍼스 내 도로망(trunk road)** 을 GNN으로 예측한다.

---

## 현재 버전: `train_v28_subgraph_gnn.ipynb`

### v28의 핵심 변경 (v25 AMP safe → v28)
단 하나의 변경: **full grid → non-building subgraph**

| 항목 | v25 | v28 |
|---|---|---|
| 그래프 노드 | 10,000개 (건물 포함) | ~5,000개 (free pixel만) |
| edge 수 | ~800,000 | ~40,000 |
| GT edge 비율 | 0.025% | ~0.5% (class imbalance 20배 개선) |

v25까지 loss가 epoch 30에서 고착되던 근본 원인 = GT edge 비율 0.025%의 극단적 class imbalance.

---

## 데이터 구조

### 입력 파일
- `collegemap/images/*_building_mask.png` — 건물 마스크 이미지
- `collegemap/txt/*_building_places.txt` — 건물 폴리곤 좌표
- `collegemap/road_masks/*_road_mask.npy` — OSM 도로 마스크 (49/86개만 존재)

### train/val/test 분리 규칙 (중요)

```
OSM road_mask.npy 있음  →  gt_source='osm'       →  train / val
OSM road_mask.npy 없음  →  gt_source='test_no_osm' →  test (GT 없음)
```

**OSM 없는 캠퍼스에는 GT를 만들면 안 된다.**
- `_generate_gt_variants()`는 OSM 있는 캠퍼스에서만 호출
- `gt_edge_mask=None`, `gt_variants=[]` 으로 저장
- 로딩 시간 낭비 및 실험 설정 오염 방지

### `load_campus()` 핵심 분기

```python
if road_map is not None:
    gt_source         = 'osm'
    gt_pixel_variants = _generate_gt_variants(terminals, is_bld, ridge, road_map)
    gt_variants       = [_make_sub_gt_edge_mask(...) for pv in gt_pixel_variants]
    gt_edge_mask      = gt_variants[0]
else:
    gt_source         = 'test_no_osm'
    gt_pixel_variants = []
    gt_variants       = []
    gt_edge_mask      = None   # ← 절대 GT 생성 안 함
```

### split 코드

```python
osm_campuses  = [c for c in campuses if c['gt_source'] == 'osm']
test_campuses = [c for c in campuses if c['gt_source'] == 'test_no_osm']
random.shuffle(osm_campuses)
n_val = max(1, len(osm_campuses)//7)
val_campuses   = osm_campuses[:n_val]
train_campuses = osm_campuses[n_val:]
```

---

## GT 생성 방식 (OSM 있는 캠퍼스, 4종)

terminals = `gate_nodes` (경계 진입점 2개) + `cluster_nodes` (건물군 대표)

| algo | 방식 | cost |
|---|---|---|
| 1 | Dijkstra | `1/(road+0.1)` — 등급 비례 선호 |
| 2 | Dijkstra | `1` — 균일 |
| 3 | Dijkstra | `1/(road²+0.01)` — 높은 등급만 강하게 |
| 4 | A* | algo1 cost + 직선 편향 |

학습 시 매 스텝 4종 중 랜덤 선택 → 데이터 증강.

---

## 노드 피처 (NODE_DIM=7, free pixel 기준)

| idx | 피처 | 설명 |
|---|---|---|
| 0 | ridge | 건물 사이 도로 ridge score |
| 1 | dist_n | 건물까지 거리 (정규화) |
| 2 | x | 픽셀 x좌표 / RES |
| 3 | y | 픽셀 y좌표 / RES |
| 4 | cluster_indicator | cluster 대표 노드면 1.0 |
| 5 | dist_to_cluster | cluster까지 exp decay |
| 6 | dist_to_gate | gate까지 exp decay |

building 피처 제거됨: 서브그래프 노드는 모두 free pixel이므로 불필요.

---

## 모델

- 3층 MHGATLayer (64dim, 4heads)
- edge classifier: `[h_s, h_d, |h_s-h_d|, h_s*h_d, raw_edge(2)]` → MLP → sigmoid
- gate_head 없음 (v25에서 제거)
- 파라미터: ~47k

## 손실 함수

- `trunk_bce_loss`: pos_weight 자동 계산 (최대 80배)
- `soft_dice_loss`: overlap 기반
- `false_positive_ridge_loss`: GT 아닌 edge 중 ridge 낮은 곳 억제
- `sparse`: `ew.mean() * 0.03`

## 학습

- AdamW, CosineAnnealingLR, AMP (`torch.amp.GradScaler('cuda')`)
- EPOCHS=220, LR=4e-4, WARMUP=20, ACCUM_STEPS=10

---

## 이전 버전 실패 이유 요약

| 버전 | 실패 원인 |
|---|---|
| v23 | diffusion loss, loss epoch 30 고착 (gt 항 18.24 불변) |
| v24 | supervised BCE 시도, full grid class imbalance |
| v25 | AMP safe, 4알고리즘 GT — 구조는 맞으나 full grid imbalance 여전 |
| v27 | hub pseudo-label, 구조 개선 — 역시 full grid |
| v28 | **현재** — non-building subgraph로 imbalance 해결 시도 중 |

---

## Claude에게 효율적으로 요청하는 법 (이 프로젝트에서 확인된 사항)

1. **"하지 말아야 할 것"을 명시적으로** — "OSM 없으면 GT 생성 함수 호출하지 마"처럼 금지 규칙을 직접 말해야 함. 기존 코드의 암묵적 제약은 잘 못 읽음.
2. **코드 전에 계획 먼저** — "어떻게 구현할 거야?" 물어보면 거기서 잡힘.
3. **"뭘 제거했어?" 체크** — 큰 파일 재작성 시 불필요한 패턴을 조용히 살려두는 경향 있음.
4. **통째로 재작성보다 타깃 수정** — 짧은 블록 단위 수정이 실수 적음.
