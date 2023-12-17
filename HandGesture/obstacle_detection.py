import os.path

import cv2
import numpy as np
import torch
from ultralytics import YOLO
import cv2

class Obstacle_detection:
    def resize_img(self, img):
        resize_ratio = 640 / int(img.shape[1])
        img_height = int(resize_ratio * int(img.shape[0]))
        re_img = cv2.resize(img, dsize=(640,img_height))

        return re_img

    def filter_boxes_by_bottom_region(self, labels, xyxys, bottom_region):
        filtered_boxes = []
        # 모든 bounding box에 대해 반복
        for label, xyxy in zip(labels, xyxys):
            x1, y1, x2, y2 = xyxy
            y2 = int(y2)

            # 만약 bounding box의 아래 부분이 아래 구역을 포함하면 선택
            if y2 >= bottom_region:
                filtered_boxes.append({'label': label, 'xyxy': xyxy})

        return filtered_boxes

    # bounding box와 라벨을 이미지에 그리는 함수
    def draw_boxes(self, image, boxes):
        for box in boxes:
            xyxy = box['xyxy']
            label = box['label']
            x1, y1, x2, y2 = xyxy
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

            # 초록색으로 bounding box 그리기
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # 라벨 텍스트 추가
            cv2.putText(image, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            cv2.putText(image, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    def detect(self, orig_image):
        # 모델 불러오기
        device = torch.device("cpu")
        model = YOLO('./model/obstacle_detection/best.pt').to(device)
        names= {0: 'bicycle', 1: 'bus', 2: 'car', 3: 'carrier', 4: 'cat', 5: 'dog', 6: 'motorcycle', 7: 'movable_signage', 8: 'person', 9: 'scooter', 10: 'stroller', 11: 'truck', 12: 'wheelchair', 13: 'barricade', 14: 'bench', 15: 'bollard', 16: 'chair', 17: 'fire_hydrant', 18: 'kioskATM', 19: 'parking_meter', 20: 'pole', 21: 'potted_plant', 22: 'power_controller', 23: 'stop', 24: 'table', 25: 'traffic_light', 26: 'traffic_light_controller', 27: 'traffic_sign', 28: 'tree_trunk'}

        # 이미지 불러와서, 가로크기 640보다 큰 이미지는 가로크기 640으로 변경하여 저장
        if orig_image.shape[1] > 640:
            image = self.resize_img(orig_image)
        else:
            image = orig_image

        # 크기 수정한 이미지로 모델학습
        result = model.predict(image)

        # 예측 결과 label과 좌표값 순서대로 저장
        label_list = []
        xyxy_list = []
        for i, re in enumerate(result[0]):
            cls = re.boxes.cls.numpy()
            xyxy = re.boxes.xyxy.numpy()
            for c, w in zip(cls,xyxy):
                c = names[int(c)]
                label_list.append(c)
                xyxy_list.append(w)

        # 이미지 높이 및 세로 3등분
        height, width = image.shape[:2]
        division_height = height // 3
        bottom_region = division_height * 2  # 이미지의 가장 아래 1/3 부분

        # 가장 아래 구역에 포함된 bounding box 선택
        filtered_boxes = self.filter_boxes_by_bottom_region(label_list, xyxy_list, bottom_region)

        # 선택된 bounding box들을 이미지에 그리기
        self.draw_boxes(image, filtered_boxes)

        # 결과를 화면에 표시
        cv2.imshow('Obstacle detection', image)
        
        return image