# SWE3032 프로젝트 컨텍스트

## 목표
캠퍼스 건물 배치 이미지 + 건물 위치 정보를 입력받아,
**캠퍼스 내 도로망(trunk road)** 을 GNN으로 예측한다.

---

## 현재 버전: `train_v28_subgraph_gnn.ipynb` (CODE_VERSION: `v28a_subgraph_gnn`)

### v28 → v28a 변경 요약

| 항목 | v28 | v28a |
|---|---|---|
| NODE_DIM | 7 | 9 (+ `dy_cluster`, `dx_cluster`) |
| 클러스터 알고리즘 | single-linkage | ball clustering (체인 방지) |
| 클러스터 분할 | 없음 | `CLUSTER_SPLIT_RADIUS=15.0` 초과 시 sub-cluster |
| GT 소스 | `road_masks/*_road_mask.npy` | `gt_masks_final/*_gt.npz` (사전 계산) |
| 데이터 증강 | 없음 | OSM 캠퍼스 4방향 flip (lr/ud/both) |
| 손실 함수 | BCE+Dice+FP_ridge+sparse | + `cluster_connectivity_loss` |
| ACCUM_STEPS | 10 | 4 |
| 체크포인트 | 단일 `.pth` | best + 50epoch마다 `_ep{N:03d}.pth` |

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
- `collegemap/gt_masks_final/*_gt.npz` — 사전 계산된 GT (49개 OSM 캠퍼스)
  - ~~`collegemap/road_masks/*_road_mask.npy`~~ — 더 이상 직접 사용 안 함

### train/val/test 분리 규칙 (중요)

```
gt_masks_final/*_gt.npz 있음  →  gt_source='osm'        →  train (4× flip aug) / val (원본)
gt_masks_final/*_gt.npz 없음  →  gt_source='test_no_osm' →  test (GT 없음)
```

**OSM GT 없는 캠퍼스에는 `gt_edge_mask=None`.**

실제 split 결과: Train 168개 (4× aug), Val 7개 (원본), Test 37개

### split 코드 (v28a)

```python
osm_orig      = [c for c in campuses if c['gt_source']=='osm' and c['name']==c['slug']]
test_campuses = [c for c in campuses if c['gt_source']=='test_no_osm']

random.seed(42); random.shuffle(osm_orig)
n_val     = max(1, len(osm_orig)//7)
val_slugs = {c['slug'] for c in osm_orig[:n_val]}

val_campuses   = [c for c in osm_orig   if c['slug'] in val_slugs]
train_campuses = [c for c in campuses   if c['gt_source']=='osm' and c['slug'] not in val_slugs]
```

---

## 클러스터 처리 (v28a 변경)

### `_cluster_centers()` — ball clustering
```python
def _cluster_centers(centers, eps=CLUSTER_EPS):
    """씨앗 기준 반경 eps 이내만 포함 (single-linkage 체인 방지)."""
```
- v28: single-linkage (체인처럼 길게 이어짐) → v28a: ball (씨앗 고정 반경)

### 클러스터 분할 (`CLUSTER_SPLIT_RADIUS=15.0`)
클러스터 max반경 > CLUSTER_SPLIT_RADIUS이면 `_cluster_centers(eps=CLUSTER_SPLIT_RADIUS)`로 재귀 분할.

---

## 데이터 증강 (v28a 신규)

OSM 캠퍼스만 4방향 flip: `[None, 'lr', 'ud', 'both']`
- `_apply_flip(arr2d, flip)` — 이미지/마스크 배열 flip
- `_flip_center(cy, cx, flip)` — 좌표 변환
- `_flip_poly(poly, W, H, flip)` — 건물 폴리곤 변환
- Val은 원본만 사용 (flip 없음)

---

## GT 생성 방식

GT는 `collegemap/gt_masks_final/*_gt.npz`에 사전 저장됨.
로딩 시 flip 적용 → `_make_sub_gt_edge_mask()`로 서브그래프 edge mask 변환.

4알고리즘으로 만든 pixel vote consensus (저장 시점에 이미 완료):

terminals = `gate_nodes` (경계 진입점 2개) + `cluster_nodes` (건물군 대표)

| algo | 방식 | cost |
|---|---|---|
| 1 | Dijkstra | `1/(road+0.1)` — 등급 비례 선호 |
| 2 | Dijkstra | `1` — 균일 |
| 3 | Dijkstra | `1/(road²+0.01)` — 높은 등급만 강하게 |
| 4 | A* | algo1 cost + 직선 편향 |

---

## 노드 피처 (NODE_DIM=9, free pixel 기준)

| idx | 피처 | 설명 |
|---|---|---|
| 0 | ridge | 건물 사이 도로 ridge score |
| 1 | dist_n | 건물까지 거리 (정규화) |
| 2 | x | 픽셀 x좌표 / RES |
| 3 | y | 픽셀 y좌표 / RES |
| 4 | cluster_indicator | cluster 대표 노드면 1.0 |
| 5 | dist_to_cluster | cluster까지 exp decay |
| 6 | dist_to_gate | gate까지 exp decay |
| 7 | dy_cluster | 가장 가까운 cluster 방향 unit vector (y) |
| 8 | dx_cluster | 가장 가까운 cluster 방향 unit vector (x) |

`_direction_to_targets(free_pixels, cluster_nodes)` → 각 free pixel에서 가장 가까운 cluster까지의 단위벡터.

---

## 모델

- 3층 MHGATLayer (64dim, 4heads)
- edge classifier: `[h_s, h_d, |h_s-h_d|, h_s*h_d, raw_edge(2)]` → MLP → sigmoid
- gate_head 없음 (v25에서 제거)
- 파라미터: ~47k

## 손실 함수 (v28a)

- `trunk_bce_loss`: pos_weight 자동 계산 (최대 80배)
- `soft_dice_loss`: overlap 기반
- `false_positive_ridge_loss`: GT 아닌 edge 중 ridge 낮은 곳 억제
- `cluster_connectivity_loss` **(v28a 신규)**: 각 terminal 노드에 `ew > EDGE_THR` edge 없으면 패널티, warmup 스케일 적용
- `sparse`: `ew.mean() * 0.03`

총 손실: `bce + 0.8*dice + fp + 0.5*conn + sparse`

## 학습

- AdamW, CosineAnnealingLR, AMP (`torch.amp.GradScaler('cuda')`)
- EPOCHS=220, LR=4e-4, WARMUP=20, ACCUM_STEPS=4
- 체크포인트: `{CODE_VERSION}.pth`, `{CODE_VERSION}_best.pth`, `{CODE_VERSION}_ep{N:03d}.pth` (50 epoch마다)

---

## 이전 버전 실패 이유 요약

| 버전 | 실패 원인 |
|---|---|
| v23 | diffusion loss, loss epoch 30 고착 (gt 항 18.24 불변) |
| v24 | supervised BCE 시도, full grid class imbalance |
| v25 | AMP safe, 4알고리즘 GT — 구조는 맞으나 full grid imbalance 여전 |
| v27 | hub pseudo-label, 구조 개선 — 역시 full grid |
| v28 | non-building subgraph로 imbalance 해결 — single-linkage 클러스터 체인 문제 |
| v28a | **현재** — ball clustering, direction feature, flip aug, connectivity loss |

---

## Claude에게 효율적으로 요청하는 법 (이 프로젝트에서 확인된 사항)

1. **"하지 말아야 할 것"을 명시적으로** — "OSM 없으면 GT 생성 함수 호출하지 마"처럼 금지 규칙을 직접 말해야 함. 기존 코드의 암묵적 제약은 잘 못 읽음.
2. **코드 전에 계획 먼저** — "어떻게 구현할 거야?" 물어보면 거기서 잡힘.
3. **"뭘 제거했어?" 체크** — 큰 파일 재작성 시 불필요한 패턴을 조용히 살려두는 경향 있음.
4. **통째로 재작성보다 타깃 수정** — 짧은 블록 단위 수정이 실수 적음.
