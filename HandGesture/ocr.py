import requests
import uuid
import json
import cv2 as cv
import pytesseract
import platform
from PIL import ImageFont, ImageDraw, Image
import numpy as np
import time

class OCR:
    def clova_ocr(self, path, config_data):
        clova_ocr_key = config_data["clova_ocr_key"]
        clova_ocr_api_url = config_data["clova_ocr_api_url"]
        files = [('file', open(path,'rb'))]
        request_json = {'images': [{'format': 'png',
                                    'name': 'output'
                                    }],
                        'requestId': str(uuid.uuid4()),
                        'version': 'V2',
                        'timestamp': int(round(time.time() * 1000))
                        }
        
        payload = {'message': json.dumps(request_json).encode('UTF-8')}
        
        headers = {
        'X-OCR-SECRET': clova_ocr_key,
        }
        
        response = requests.request("POST", clova_ocr_api_url, headers=headers, data=payload, files=files)
        
        result = response.json()
        img = cv.imread(path)
        roi_img = img.copy()
        
        
        
        for field in result['images'][0]['fields']:
            text = field['inferText']
            vertices_list = field['boundingPoly']['vertices']
            pts = [tuple(vertice.values()) for vertice in vertices_list]
            topLeft = [int(_) for _ in pts[0]]
            topRight = [int(_) for _ in pts[1]]
            bottomRight = [int(_) for _ in pts[2]]
            bottomLeft = [int(_) for _ in pts[3]]
        
            cv.line(roi_img, topLeft, topRight, (0,255,0), 2)
            cv.line(roi_img, topRight, bottomRight, (0,255,0), 2)
            cv.line(roi_img, bottomRight, bottomLeft, (0,255,0), 2)
            cv.line(roi_img, bottomLeft, topLeft, (0,255,0), 2)
            roi_img = OCR.put_text(roi_img, text, topLeft[0], topLeft[1] - 10, font_size=30)
        
        return roi_img, text
    
    def pyte(self, cropped_image):
        if platform.system() == 'Darwin':
            pytesseract.pytesseract.tesseract_cmd = r''
        elif platform.system() == 'Windows':
            pytesseract.pytesseract.tesseract_cmd = r'C:/Program Files/Tesseract-OCR/tesseract'
        text = pytesseract.image_to_string(cropped_image, lang='kor+eng')
        cv.putText(cropped_image, text, (10, 30), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, cv.LINE_AA)
        return cropped_image, text
        
    def put_text(self, image, text, x, y, color=(0, 255, 0), font_size=22):
        if type(image) == np.ndarray:
            color_coverted = cv.cvtColor(image, cv.COLOR_BGR2RGB)
            image = Image.fromarray(color_coverted)
    
        if platform.system() == 'Darwin':
            font = 'AppleGothic.ttf'
        elif platform.system() == 'Windows':
            font = 'malgun.ttf'
            
        image_font = ImageFont.truetype(font, font_size)
        font = ImageFont.load_default()
        draw = ImageDraw.Draw(image)
    
        draw.text((x, y), text, font=image_font, fill=color)
        
        numpy_image = np.array(image)
        opencv_image = cv.cvtColor(numpy_image, cv.COLOR_RGB2BGR)
    
        return opencv_image
