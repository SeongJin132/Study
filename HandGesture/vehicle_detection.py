import cv2
from ultralytics import YOLO
from gtts import gTTS
import os
import threading
from playsound import playsound

class Vehicle_detection:
    def play_sound(self, detected_info):
        # Text-to-Speech
        tts = gTTS(text=detected_info, lang='ko')
        tts.save("detected_info.mp3")
        playsound("detected_info.mp3")

    def detect(self, last_detected, t, model, frame):
        results = model(frame)  # Run YOLOv8 inference on the frame
        target_objects = ['bicycle', 'bus', 'car', 'motorcycle', 'scooter', 'truck']
        for result in results:
            for cls, box in zip(result.boxes.cls, result.boxes.xyxy):
                # Get the class name
                class_name = result.names[int(cls.item())]
                
                # Check if the detected object is in the target objects
                if class_name in target_objects:
                    # Check if the detected object is in the bottom 3 grids
                    _, _, x_center, y_center = box
                    height, width = frame.shape[:2]
                    if y_center > 2 * height / 3:
                        print(class_name)
                        print(last_detected)
                        if class_name != last_detected:
                            last_detected = class_name
                            detected_info = f"{class_name}이 앞에 있습니다."
                            print(detected_info)

                            if t is None or not t.is_alive():
                                t = threading.Thread(target=self.play_sound, args=(detected_info,))
                                t.start()
                        else:
                            continue
        annotated_frame = results[0].plot()  # Visualize the results on the frame
        cv2.imshow('Obstacle detection', annotated_frame)
        return last_detected, t, annotated_frame