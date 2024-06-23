# AI 기술을 활용한 HTP 검사 자동화 시스템 개발

## 1. 서론
- 미술치료: 예술적 창조 행위가 치료의 한 방법
- 그림검사: 아동의 무의식을 반영, 솔직한 대답 유도

## 2. HTP 그림 검사
- 집 나무 사람 그림 검사는 각각 요소의 특징을 분석해 심리 파악
- 사람 그림 검사는 그림 파악을 통해 파악하기 힘들고 부정적 요소를 추가할 가능성이 있기에 제외.

## 3. 기존 연구 검토
- Faster-R-CNN: Object Detection으로 아동 그림 심리 분석
- YOLO: 빠른 검출 속도와 높은 정확도
- CNN: 퍼지 추론을 통한 맞춤 분석

## 4. 데이터 셋 소개
- AIHUB 기반 아동 미술심리 진단 그림 데이터 구축

## 5. 문제점들과 해결법
- 추상적 평가항목에 대한 모델 인식 어려움
- 전문가 상담을 추가하여 해결

## 6. 이미지 데이터 학습 과정
- YoloV5, YoloV8M을 사용한 이미지 데이터 학습

## 7. 학습 결과 해석
- 학습 진행에 따른 손실함수 값 분석

## 8. 모델 성능 테스트
- 다양한 테스트 이미지로 모델 성능 평가

## 9. Chat bot을 통한 상담
- 전문가 상담 필요한 항목에 대해 추가 검사
- 추가 라벨링, GPT 프롬프트 제작, 정확도 개선 필요

## 활용방안
- 그림검사와 상담결과를 통합하여 비교적 간단하게 최종 결과 도출
