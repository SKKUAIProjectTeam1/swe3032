# v28 세 모델 비교 (GAT / MLP / SAGE)

v28 subgraph 파이프라인에서 모델 구조만 바꿔가며 실험한 세 노트북의 차이를 정리한 문서.

---

## 공통 파이프라인

세 노트북은 모델 아키텍처 부분을 제외하고 동일한 파이프라인을 공유한다.

- 서브그래프 노드 추출: full grid(10,000개) 대신 건물 없는 free pixel만 사용 (~9,000개), class imbalance 20배 개선
- 노드 피처 9차원: ridge, dist_n, x, y, cluster_indicator, dist_to_cluster, dist_to_gate, dy_cluster, dx_cluster
- 손실 함수: BCE + Soft Dice + FP Ridge + Cluster Connectivity + Sparse Loss
- 학습: AdamW, CosineAnnealingLR, PyTorch AMP, 220 epoch

---

## 모델 비교

| 항목 | GAT (`train_v28_subgraph_gnn.ipynb`) | MLP (`train_v28_subgraph_mlp.ipynb`) | SAGE (`train_v28_subgraph_sage.ipynb`) |
| :--- | :--- | :--- | :--- |
| 모델명 | `SubgraphGAT` | `SubgraphMLP` | `SubgraphSAGE` |
| 핵심 레이어 | `MHGATLayer` (4-head attention) | `nn.Linear` (메시지 패싱 없음) | `SAGEConv` (mean aggregation) |
| 이웃 정보 활용 | attention 가중 집계 | 없음 | 평균 집계 후 concat |
| 파라미터 수 | 47,425 | 46,657 | 55,425 |
| 역할 | 메인 모델 | 그래프 구조 없는 대조군 | attention 없는 GNN 대조군 |

---

## 아키텍처 상세

### GAT

- `MHGATLayer` 3층 (64차원, 4 head)
- 이웃 노드와의 관계에 따라 동적으로 attention 가중치 계산 후 집계
- 에지 분류기: `[h_src, h_dst, |h_src-h_dst|, h_src*h_dst, edge_ridge(2)]` → MLP → sigmoid

### MLP

- 3층 Linear 블록 (Linear + LayerNorm + ELU + Dropout)
- 메시지 패싱 없이 노드 피처만으로 노드 표현 생성
- "피처 자체의 힘이 얼마나 되는가"를 측정하는 하한선 역할

### SAGE

- `SAGEConv` 3층
- 이웃 평균값과 자기 피처를 concat 후 선형 변환
- attention 연산 없이 단순 mean aggregation → GAT 대비 연산량 적음

---

## 분석 관점

- **GAT vs MLP**: 그래프 구조(메시지 패싱) 유무가 도로 연결성에 얼마나 기여하는가
- **GAT vs SAGE**: attention 기반 동적 가중치와 단순 평균 집계 중 어느 쪽이 캠퍼스 도로 패턴을 더 잘 잡는가
