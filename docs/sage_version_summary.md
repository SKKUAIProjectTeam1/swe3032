# GraphSAGE 비교군 구현 노트 (train_v28_subgraph_sage.ipynb)

GAT/MLP와의 비교를 위해 `train_v28_subgraph_gnn.ipynb`를 베이스로 GraphSAGE 버전을 별도 구현했다.

## 변경 내용

**아키텍처 교체**
- `MHGATLayer` 제거, `SAGEConv` (Mean Aggregation) + `SubgraphSAGE`로 대체
- PyG 등 외부 라이브러리 없이 순수 PyTorch로 직접 구현해 기존 파이프라인과 호환 유지
- SAGEConv 동작: 이웃 노드 피처를 degree로 나눠 평균 집계 → 자기 피처와 concat → Linear + ELU + LayerNorm + Dropout
- 3층으로 쌓아 `SubgraphSAGE` 구성, 파라미터 수 ~55k

**공통 유지 항목**
- 데이터 로딩·전처리·증강 파이프라인 동일
- 손실 함수(BCE + Dice + FP + Connectivity + Sparse) 동일
- 학습 설정(AdamW, CosineAnnealing, 220 epoch, AMP) 동일
- `CODE_VERSION = 'v28a_subgraph_sage'`으로 체크포인트 파일명 분리

## 목적

Attention 기반(GAT)과 단순 평균 집계(SAGE) 중 캠퍼스 도로망 패턴 학습에 더 적합한 구조가 무엇인지 비교하기 위한 모델이다.
