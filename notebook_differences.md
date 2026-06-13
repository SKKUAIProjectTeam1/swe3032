# v28 Subgraph 노트북별 모델 비교 분석 (GNN vs MLP vs SAGE)

본 문서는 캠퍼스 도로망(trunk road) 예측 프로젝트의 v28 서브그래프(subgraph) 학습 파이프라인에서 독립적으로 공존하는 세 가지 노트북(`gnn`, `mlp`, `sage`)의 아키텍처 및 모델링 차이점을 설명합니다.

---

## 1. 개요 및 공통 파이프라인

세 노트북은 모델 아키텍처 부분을 제외하고 다음의 데이터 처리 및 학습 파이프라인을 완전히 공유합니다.

* **서브그래프 노드 추출**: 기존 full grid(10,000개 노드) 대비 건물이 없는 영역의 free pixel만을 추출하여 노드 수를 약 5,000개로 축소하고 클래스 불균형(Class Imbalance)을 20배 개선.
* **노드 피처 (NODE_DIM=9)**: 
  1. `ridge`: 건물 사이 도로 ridge score
  2. `dist_n`: 건물까지의 정규화 거리
  3. `x`: x 좌표 / RES
  4. `y`: y 좌표 / RES
  5. `cluster_indicator`: 클러스터 대표 노드 여부
  6. `dist_to_cluster`: 클러스터까지의 지수 감쇄 거리
  7. `dist_to_gate`: 게이트(진입점)까지의 지수 감쇄 거리
  8. `dy_cluster`: 가장 가까운 클러스터 방향 단위 벡터 (y)
  9. `dx_cluster`: 가장 가까운 클러스터 방향 단위 벡터 (x)
* **손실 함수**: BCE + Soft Dice + FP Ridge + Cluster Connectivity + Sparse Loss의 가중합 조합 사용.
* **학습 파이프라인**: AdamW, CosineAnnealingLR, PyTorch AMP(자동 혼합 정밀도), 220 Epoch, Accumulation steps 적용.

---

## 2. 모델별 비교 요약

| 항목 | [GNN (GAT)](./train_v28_subgraph_gnn.ipynb) | [MLP](./train_v28_subgraph_mlp.ipynb) | [SAGE](./train_v28_subgraph_sage.ipynb) |
| :--- | :--- | :--- | :--- |
| **모델명** | `SubgraphGAT` | `SubgraphMLP` | `SubgraphSAGE` |
| **적용 레이어** | `MHGATLayer` (Multi-Head Attention) | `nn.Linear` (단순 밀집 레이어) | `SAGEConv` (Mean Aggregation) |
| **메시지 패싱** | **Attention 기반** 이웃 정보 가중 결합 | **없음** (노드 단위 개별 연산) | **이웃 평균 정보** + 자기 피처 결합 |
| **파라미터 수** | 47,425개 | 46,657개 | 55,425개 |
| **역할 및 의의** | 동적 관계 포착이 가능한 기본 GNN 모델 | 그래프 구조 정보를 사용하지 않는 성능 대조군 | 단순 Aggregation의 효율성 검증 대안 모델 |

---

## 3. 모델 아키텍처 세부 비교

### ① Graph Attention Network (GAT)
* **파일**: `train_v28_subgraph_gnn.ipynb`
* **핵심 레이어**: `MHGATLayer` (4 Heads, 64차원 출력)
* **동작 원리**: 
  - 각 노드 피처를 선형 변환 후, 이웃 노드와의 기하학적/위상학적 관계에 따른 Attention score를 계산합니다.
  - Softmax 함수를 통해 계산된 동적 가중치로 이웃 노드의 메시지를 취합(Aggregation)하여 새로운 노드 표현을 생성합니다.
* **특징**: 복잡하고 유연한 노드 간의 위상학적 관계 학습에 적합하지만, Attention 가중치를 매번 계산해야 하므로 상대적으로 연산 복잡도가 큽니다.

### ② Multi-Layer Perceptron (MLP)
* **파일**: `train_v28_subgraph_mlp.ipynb`
* **핵심 레이어**: 3층의 선형 레이어 블록 (`nn.Linear` + `LayerNorm` + `ELU` + `Dropout`)
* **동작 원리**:
  - 그래프 상에서 노드 간의 연결(Edge) 정보를 통한 정보 전파(Message Passing)를 일체 수행하지 않습니다.
  - 주어진 노드 피처(`NODE_DIM=9`) 자체만을 MLP 레이어를 통해 독립적으로 변환하여 노드 표현을 만듭니다.
* **특징**: 이 모델의 성능은 **"그래프 위상 구조 정보 없이, 단독 피처(위치, 게이트/클러스터 거리 등)만을 사용했을 때 성능이 얼마나 나오는가?"**를 평가하는 **Baseline 대조군** 역할을 합니다. 

### ③ GraphSAGE (SAGE)
* **파일**: `train_v28_subgraph_sage.ipynb`
* **핵심 레이어**: `SAGEConv` (Mean Aggregation 및 Concatenation)
* **동작 원리**:
  - 연결된 이웃 노드들의 피처를 수집하여 산술 평균(Mean)을 냅니다.
  - 자기 자신의 피처 정보와 이웃 노드의 평균 취합 정보를 가로로 병합(`torch.cat([x, agg_mean], dim=1)`)한 후, 선형 변환과 활성화 함수를 통과시킵니다.
* **특징**: 동적인 가중치를 매번 계산하는 GAT와 달리 평균값을 사용하여 직관적이고 연산량이 적으며, 자기 자신과 이웃의 정보가 구분(Concatenate)되어 업데이트되므로 구조 파악 성능이 강력합니다.

---

## 4. 분석 가이드라인

성능(BCE Loss, Dice Coefficient 등) 및 시각화 결과 분석 시 다음 관점을 중심적으로 비교하는 것을 권장합니다.
1. **GNN vs MLP**: 그래프 정보(Message Passing)의 유무가 도로망 연결성 및 끊김(Disconnectivity)을 해결하는 데 얼마나 기여하는가?
2. **GAT vs SAGE**: 복잡한 동적 가중치(Attention) 메커니즘을 적용한 것과 단순 평균(SAGE Mean)을 취한 모델 중 어떤 것이 캠퍼스 도로의 공간적 패턴을 더 안정적으로 추출하는가?
