import cv2
import mediapipe as mp
import numpy as np
import keyboard
import time
from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
import pyautogui
from pywinauto.application import Application
### 새로 추가된 라이브러리 ###
from collections import Counter
from konlpy.tag import Okt  #  pip install konlpy + JAVA JDK21 검색 후 다운 필수
import os
from gtts import gTTS 
from playsound import playsound
import pygame
import platform



class google_img_search:
    # 구글 이미지 검색 함수 정의
    def search_google_images(self, Path):
        Path = os.path.join(os.getcwd(), Path)
        ## 창이 닫히는 현상을 방지하고, 크롬드라이버 실행
        chrome_options = Options()
        chrome_options.add_experimental_option('detach', True)
        driver = webdriver.Chrome(options=chrome_options)
        
        
        # 구글 이미지 검색 페이지 열기
        driver.get("https://www.google.com/imghp")
        # driver.minimize_window()
        
        # 이미지 업로드 버튼 찾기
        upload_button = driver.find_element(By.CLASS_NAME, "Gdd5U")
        
        # 이미지 업로드 버튼 클릭
        upload_button.click()
        
        # 파일 업로드 창 열기
        upload_input = driver.find_element(By.CLASS_NAME, "DV7the")
        time.sleep(0.5)
        upload_input.click()
        time.sleep(1)
        
        
        # 크롭한 이미지 파일을 검색하기.
        pyautogui.typewrite(Path)
        pyautogui.press('enter')
        time.sleep(1.5)
        
        """
        검색 결과에서 객체 이름 가지고 오기
        """
        
        # 검색 결과로 나온 text 긁어오기.
        all_text = driver.find_elements(By.CLASS_NAME, 'UAiK1e')
        text_list = []
        if all_text :
            for i,element in enumerate(all_text):
                text = element.text
                text = text.split("|")[0].split(':')[0].strip()
                text = text.split(" ")
                text_list.append(text)
                            
            print(text_list)       
            
        else :
            no_img = '검색된 이미지가 없습니다. 대상을 다시 찍어주시겠어요?'
            print(no_img)
            
            # 음성으로 이름 읽어주기.
            try_again = './mp3/try_again.mp3'
            tts_ko = gTTS(text = no_img , lang = 'ko')
            tts_ko.save(try_again)
            pygame.mixer.init()
            pygame.mixer.music.load(try_again)
            pygame.mixer.music.play() # 재생하기.
            
            # 재생하는 동안 유지.
            while pygame.mixer.music.get_busy():
                pygame.time.Clock().tick(10)
                break   
            
        if text_list is not None :
            message = "이미지를 분석 중입니다. 잠시만요."
                            
            # 음성으로 이름 읽어주기.
            waiting = './mp3/wait.mp3'
            tts_ko = gTTS(text = message , lang = 'ko')
            tts_ko.save(waiting)
            # playsound(waiting)
            pygame.mixer.init()
            pygame.mixer.music.load(waiting)
            pygame.mixer.music.play() # 재생하기.
            
            # 재생하는 동안 유지.
            while pygame.mixer.music.get_busy():
                pygame.time.Clock().tick(10)
            
        # print(text_list)
        
    
            
        # JAVA JDK 환경설정
        # JDK 다운 폴더 경로를 지정함.
        if platform.system() == 'Darwin':
            print("")
        elif platform.system() == 'Windows':
            os.environ['JAVA_HOME'] = r'C:\\Program Files\\Java\\jdk-21\\bin'
         
        okt = Okt()  #Okt 함수를 지정함. 
        
        merged_sentence = ' '.join(' '.join(sublist) for sublist in text_list)
        nouns = okt.nouns(merged_sentence) # list에 들어 있던 text를 한 줄로 만들어서 형태소만 파악하기.
        
        # 명사의 종류와 빈도수 나타내기.
        counter_nouns = Counter(nouns)
        predict_name = counter_nouns.most_common(12) # 가장 많이 나온 단어와 빈도
        print(predict_name)
        
        it_is_name = predict_name[0][0] # 단어만 반환하기.
        print(it_is_name)
        
        
        # 음성으로 이름 읽어주기.
        name_tts = './mp3/name.mp3'
        tts_ko = gTTS(text = "선택하신 것은" + it_is_name + "입니다." , lang = 'ko')
        tts_ko.save(name_tts)
        pygame.mixer.init()
        pygame.mixer.music.load(name_tts)
        pygame.mixer.music.play() # 재생하기.
            
        # 재생하는 동안 유지.
        while pygame.mixer.music.get_busy():
            pygame.time.Clock().tick(10)
        # playsound(name_tts)