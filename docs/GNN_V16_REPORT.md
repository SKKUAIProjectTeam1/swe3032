# 🏆 GNN 기반 자율 도로 설계 최종 보고서 (V16)
**"Neural Civil Engineer: Ridge-Aligned Structural Learning"**

## 1. 프로젝트 요약
본 프로젝트는 성균관대학교 자연과학캠퍼스의 건물 배치를 입력으로 받아, 인간의 개입(수업 데이터, 기존 도로 정보) 없이 **순수 GNN(Graph Neural Network)**이 기하학적 효율성과 물리적 타당성을 갖춘 최적의 도로망을 스스로 설계하도록 학습시켰습니다.

## 2. V16 모델의 핵심 혁신 (The Civil Engineer)
이전 버전의 '따개비(건물 밀착)' 및 '단절' 문제를 해결하기 위해 다음과 같은 고급 GNN 최적화 기법을 적용했습니다.

- **능선 보상 (Ridge/Medial Axis Reward):** 
  - 건물 사이의 정중앙 지점(Gap)을 찾는 **Distance Transform** 기법을 적용.
  - 도로가 건물에 붙지 않고, 건물 사이의 '골짜기'를 따라 흐르도록 유도하여 현실적인 이격 거리 확보.
- **차수 제어 (Degree Regularization):**
  - 각 노드의 연결(Edge) 수를 2~3개로 제한하는 패널티 부여.
  - 면(Surface) 형태의 흩뿌려진 길이 아닌, 선명하고 깔끔한 **간선 도로(Strands)** 유도.
- **고해상도 위상 학습 (100x100 Grid):**
  - 10,000개의 노드와 80,000개의 엣지 위에서 메세지 패싱을 수행.
  - 50,000 Step의 고강도 학습을 통해 캠퍼스 전체를 아우르는 전역적 연결성 확보.
- **자율 관문 최적화:**
  - 내부 도로망의 흐름과 연동하여 정문(MAIN GATE)과 후문(BACK GATE) 위치를 기하학적으로 도출.

## 3. 기술적 사양
- **Framework:** PyTorch & PyTorch Geometric
- **Resolution:** 100 x 100 Mesh Graph
- **Optimization:** Adam Optimizer (50,000 steps)
- **Constraints:** Zero-Collision (Hard Penalty), Perimeter Ridge Preference, Global Connectivity Message Passing.

## 4. 최종 결과물
- **생성된 지도:** `campus_v16_civil_design.png`
- **주요 성과:** 
  1. 85동과 21~27동 사이의 끊김 없는 유기적 연결로 확보.
  2. 건물을 관통하지 않으면서도 최단 거리로 순환하는 고리(Loop) 구조 형성.
  3. 실제 성대 자과캠 도로망과 비교했을 때 기하학적으로 더 효율적인 '대안 경로' 제시.

## 5. 결론
V16 모델은 AI가 단순한 데이터 분류기를 넘어, **물리적 제약 조건과 기하학적 목적 함수를 이해하고 스스로 인프라를 설계할 수 있음**을 증명하였습니다. 이는 향후 스마트 시티 설계 및 자율주행 도로 최적화 분야에 응용될 수 있는 강력한 기초 모델이 될 것입니다.

---
**최종 업데이트:** 2026년 6월 7일
