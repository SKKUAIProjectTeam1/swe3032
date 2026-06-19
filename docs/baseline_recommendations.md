# 추가 베이스라인 검토 메모

GAT / MLP / SAGE 외에 이 태스크(subgraph 엣지 분류)에 적용 가능한 베이스라인 후보들을 정리.
시간 관계상 실제 구현은 못 했고, 향후 확장 시 참고용.

---

## 1. 트리 기반 모델 (LightGBM / XGBoost)

subgraph 엣지를 테이블 행으로 펼치면 정형 분류 문제가 된다.

피처 구성:
- 소스/목적 노드 피처 각 9차원 → 18차원
- 두 피처의 차이(절댓값), element-wise 곱
- 엣지 길이(유클리드), edge_ridge 점수

가장 빠르고(몇 초 이내) feature importance로 어떤 피처가 중요한지 바로 확인 가능하다.
GNN 없이도 정형 ML이 얼마나 되는지 보는 lower bound 역할을 할 수 있다.

---

## 2. GCN (Graph Convolutional Network)

GAT에서 attention을 빼고 degree 기반 가중평균만 쓰는 단순화 버전.

$$H^{(l+1)} = \sigma(\tilde{D}^{-1/2} \tilde{A} \tilde{D}^{-1/2} H^{(l)} W^{(l)})$$

`MHGATLayer` 대신 `scatter_add_`로 구현하면 파라미터 수와 연산량이 줄고, attention 과적합 위험도 낮아진다.
격자 구조 그래프에서 GAT와 GCN 중 어느 쪽이 유리한지 보는 데 의미가 있다.

---

## 3. 휴리스틱 (Dijkstra / A*)

학습 없이 ridge 점수를 비용으로 써서 터미널(게이트 + 건물군) 간 최단 경로를 연결하는 방식.

- 엣지 비용: `1.0 / (edge_ridge + ε)` 또는 `dist_to_building` 반영
- GT 없어도 즉시 적용 가능 → test 캠퍼스 시각화에 유용
- "학습 모델이 단순 탐색보다 얼마나 낫냐"를 보는 기준선

실제로 MST/Ridge-only 베이스라인으로 이미 구현돼 있고 (`scripts/baseline_eval.py`), 비슷한 방향이다.

---

## 4. 좌표 기반 MLP (Coordinate-based)

ridge, gate 거리 등 도메인 피처를 전부 무시하고 좌표(x_src, y_src, x_dst, y_dst)만으로 엣지 확률을 예측하는 모델.

캠퍼스마다 도로 위치에 공간적 편향이 있다면 좌표만 봐도 어느 정도 맞출 수 있다. 이게 잘 나오면 모델이 진짜 구조를 학습한 게 아니라 위치를 암기하는 것일 수 있어서, 우리가 쓰는 피처가 실제로 의미 있는지 검증하는 데 쓸 수 있다.
