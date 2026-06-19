# MLP 비교군 구현 노트 (train_v28_subgraph_mlp.ipynb)

GAT와의 공정한 비교를 위해 `train_v28_subgraph_gnn.ipynb`를 베이스로 MLP 버전을 별도 구현했다.

## 변경 내용

**아키텍처 교체**
- `MHGATLayer` 전체 제거, `SubgraphMLP`로 대체
- 메시지 패싱 없이 노드 피처(9차원)만으로 엣지 분류
- 파라미터 수는 GAT/SAGE와 최대한 맞춰 조건 통일 (~47k)

**공통 유지 항목**
- 데이터 로딩·전처리·증강 파이프라인 동일
- 손실 함수(BCE + Dice + FP + Connectivity + Sparse) 동일
- 학습 설정(AdamW, CosineAnnealing, 220 epoch, AMP) 동일
- `CODE_VERSION = 'v28a_subgraph_mlp'`으로 체크포인트 파일명 분리

## 목적

그래프 구조 정보(이웃 노드 메시지)가 실제로 성능에 기여하는지 확인하기 위한 대조군이다.
MLP가 GAT와 비슷한 성능을 낸다면, 모델 구조보다 노드 피처(ridge, gate 거리 등) 자체의 영향이 크다는 의미다.
