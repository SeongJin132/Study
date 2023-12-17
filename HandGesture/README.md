# requirements
- mediapipe 0.8.4
- OpenCV 4.6.0.66 or Later
- Tensorflow 2.9.0 or Later
- protobuf <3.20,>=3.9.2
- scikit-learn 1.0.2 or Later
- matplotlib 3.5.1 or Later
# Model Downloads
- model/obstacle_detection/best.pt : https://1drv.ms/u/s!Aic8Abrt7GADl9VQlXK6vpTLqPkh4Q?e=VrsQS8
- model/vidio/best.pt : https://1drv.ms/u/s!Aic8Abrt7GADl9ZRS4WljXpjX0KprA?e=J6LnFX

# app.py 실행 시
1. 미리 학습된 손모양 데이터 모델로 불러옴
2. 연결된 웹캠 감지
3. mediapipe로 손 객체 감지
4. 손모양이 특정 프레임동안 같은 손모양이 감지되었을 경우
     1. 손모양이 hold 이면 중지, 검지를 기준으로 이미지가 저장됨
          - google_img_search 작동
          - ![google_img_search_demo](https://github.com/Adihang/hand-gesture-recognition-using-mediapipe/assets/56463432/78af6ce6-3256-4fdf-bfdd-94048bf63a64)
     2. 손모양이 one 이면 화면에서 손이 사라졌을때 이미지가 저장됨
          - Vehicle_detection 작동
          - ![Vehicle_detection_demo (1)](https://github.com/Adihang/hand-gesture-recognition-using-mediapipe/assets/56463432/80294e57-e36e-4271-ba03-6b0f621ee97b)


# 학습 데이터 수집방법
1. app.py 실행
2. 데이터 라벨 명 수정
     - 'model/keypoint_classifier/keypoint_classifier_label.csv'
     - 'model/point_history_classifier/point_history_classifier_label.csv'
3. 수집할 데이터 선택
     - A key - standby mode
     - S key - keypoint 0~9 mode
     - D key - keypoint 10~19 mode
     - F key - pointhistory mode
     - 0~9 key - label select
4. mode와 label이 선택되면 프레임마다 데이터로 수집됨
     - 선택이 되면 바로 데이터 수집이 시작되니 미리 손모양을 만들어놓고 선택할 것
5. A를 입력하면 다시 standby mode로 데이터 수집이 정지됨

# 수집한 데이터 학습방법
1. keypoint_classification.ipynb (손 모양 학습)
2. point_history_classification.ipynb (손 동작 학습)
