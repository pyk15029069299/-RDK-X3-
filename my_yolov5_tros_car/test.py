from PyQt5.QtWidgets import QApplication, QMainWindow, QTextEdit,QLabel
from PyQt5 import QtGui
from PyQt5.uic import loadUi
from PyQt5.QtCore import Qt,QTimer,pyqtSignal
import sys
from hobot_vio import libsrcampy as srcampy
import cv2
import numpy as np
import dlib
import pyttsx3
from time import time
import math 
from USBCAM import *
import subprocess
import os
from my_code4 import ImageSubscriber
import rclpy
from rclpy.node import Node
from ai_msgs.msg import PerceptionTargets
from sensor_msgs.msg import Image
from cv_bridge import CvBridge


class MainWindow(QMainWindow):
    def __init__(self):
        super(MainWindow, self).__init__()
        # Load the UI file
        loadUi("untitled.ui", self)
        self.pushButton.setStyleSheet("background-image: url(background1.jpg);")
        self.groupBox_1.setStyleSheet("QGroupBox { border: 2px solid white; border-color: black }")
        self.groupBox_2.setStyleSheet("QGroupBox { border: 2px solid white; border-color: black }")

        self.text_edit = QTextEdit(self.centralwidget)
        self.text_edit.setGeometry(530, 70, 191, 341)
        self.text_edit.setReadOnly(True)  # 设置为只读
        self.text_edit.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)  # 设置垂直滚动条根据需要显示
        
        self.label_img = QLabel(self.groupBox_1)
        self.label_img.setGeometry(10,30,400,300)
        self.label_img.setAlignment(Qt.AlignTop|Qt.AlignLeft)
        
        self.pushButton_1.clicked.connect(self.open_camera1)
        self.pushButton_2.clicked.connect(self.shut_camera1)
        
        self.timer=QTimer(self)
        self.timer.timeout.connect(self.update_camera)
        self.timer.start(1000//30) 
        
        self.PREDICTOR_PATH = "shape_predictor_68_face_landmarks.dat"
        self.detector = dlib.get_frontal_face_detector()  #创建人脸检测对象
        self.predictor = dlib.shape_predictor(self.PREDICTOR_PATH) #创建人脸关键点预测器对象
        
        self.engine = pyttsx3.init()
        self.count_ear = 0
        self.count_mar = 0
        self.frame_counter = 0
        self.check_interval = 25
        self.is_blinking = False
        self.is_yawning = False
        
        self.pushButton_3.clicked.connect(self.open_tros)
        self.label_img2 = QLabel(self.groupBox_2)
        self.label_img2.setGeometry(10,30,400,300)
        self.label_img2.setAlignment(Qt.AlignTop|Qt.AlignLeft)    
        self.pushButton_4.clicked.connect(self.close_tros)
        self.tros_activate = False

        rclpy.init()
        self.image_subscriber = ImageSubscriber()                  

    def open_camera1(self):
        # Append new text to text_edit
        #current_text = self.text_edit.toPlainText()
        #new_text = "打开摄像头1\n"
        #self.text_edit.setPlainText(current_text + new_text)
        self.cam = srcampy.Camera()
        self.cam.open_cam(0,-1,1,672,672)
    
    def update_camera(self):      
        if hasattr(self,'cam'):
            img = self.cam.get_img(2,672,672)
            if img is not None:
                frame = np.frombuffer(img,dtype=np.uint8)
                frame = frame.reshape(672*3//2,672)
                img_bgr = cv2.cvtColor(frame,cv2.COLOR_YUV2BGR_NV12)
                resized_img_bgr = cv2.resize(img_bgr, (320,240), interpolation=cv2.INTER_AREA)
                
                
                gray_image = cv2.cvtColor(resized_img_bgr, cv2.COLOR_BGR2GRAY)
                rects = self.detector(gray_image)

                for face in rects:
                    landmarks = self.predictor(gray_image, face)
                    for n in range(0, 68):
                        x = landmarks.part(n).x  
                        y = landmarks.part(n).y 
                    landmarks = shape_to_np(landmarks)    
                    leftEye = landmarks[42:48]
                    rightEye = landmarks[36:42]
                    mouth = landmarks[48:68]
                    leftEAR = eye_aspect_ratio(leftEye)
                    rightEAR = eye_aspect_ratio(rightEye)
                    ear = (leftEAR + rightEAR) / 2.0
                    mar = mouth_aspect_ratio(mouth)
                    leftEyeHull = cv2.convexHull(leftEye)
                    rightEyeHull = cv2.convexHull(rightEye)
                    mouthHull = cv2.convexHull(mouth)
                    cv2.drawContours(resized_img_bgr, [leftEyeHull], -1, (0, 255, 0), 1)                    
                    cv2.drawContours(resized_img_bgr, [rightEyeHull], -1, (0, 255, 0), 1)                    
                    cv2.drawContours(resized_img_bgr, [mouthHull], -1, (0, 255, 0), 1)
                    print(ear)
                    if ear < 0.2:
                        self.is_blinking =True
                        self.count_ear +=1
                    if mar >0.9:
                        self.is_yawning = True 
                        self.count_mar +=1
                        
                    self.frame_counter +=1
                    print(self.frame_counter)
                    if self.frame_counter >= self.check_interval:
                        self.frame_counter=0
                        if self.is_blinking or self.is_yawning:
                            print(self.is_blinking)  
                            print(self.is_yawning)
                            #current_text = self.text_edit.toPlainText()                       
                            #lines = current_text.split('\n')
                            #if len(lines)>=1:
                            #    lines[1] = f"眨眼:{self.count_ear},yawning:{self.count_mar}"
                            #else:
                            #    lines.insert(1,f"眨眼:{self.count_ear},yawning:{self.count_mar}")
                            #new_text = "\n".join(lines)
                            #self.text_edit.setPlainText(new_text)
                            if  self.count_ear>10 or self.count_mar>5:
                                volume = self.engine.getProperty('volume')
                                rate = self.engine.getProperty('rate')
                                self.engine.setProperty('volume',10)
                                self.engine.setProperty('rate',rate-100)
                                self.engine.say("Please rest")
                                self.engine.runAndWait()
                                self.count_ear = 0
                                self.count_mar = 0                    

                    
                img_rgb = cv2.cvtColor(resized_img_bgr,cv2.COLOR_BGR2RGB)
                h,w,ch = img_rgb.shape
                bytes_per_line = ch*w
                q_img = QtGui.QImage(img_rgb.data,w,h,bytes_per_line,QtGui.QImage.Format_RGB888)
                pixmap = QtGui.QPixmap.fromImage(q_img)
            self.label_img.setPixmap(pixmap)
            
            
    def shut_camera1(self):
        # self.timer.stop()
        if hasattr(self,'cam'):
            self.cam.close_cam()
            del self.cam
            self.label_img.clear()


    def open_tros(self):
        image_subscriber = self.image_subscriber
        self.tros_activate = True
        while self.tros_activate:
            rclpy.spin_once(image_subscriber)
            image = image_subscriber.get_image()
            msg = image_subscriber.get_msg()  
            if msg[1]:
                print(msg) 
                current_text = self.text_edit.toPlainText()
                new_text = "fps:"+ str(msg[0]) + " class:" +str(msg[1]) + " conf:" + str(round(msg[2],2)) + "\n"
                self.text_edit.setPlainText(current_text + new_text)
            if image is not None:
                cv2.imshow('my_image',image)
                image = cv2.resize(image,(320,240))
                image_rgb = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
                h,w,ch = image_rgb.shape
                bytes_per_line = ch*w
                q_image = QtGui.QImage(image_rgb.data,w,h,bytes_per_line,QtGui.QImage.Format_RGB888)
                pixmap = QtGui.QPixmap.fromImage(q_image)
                self.label_img2.setPixmap(pixmap)
                QApplication.processEvents()

        #image_subscriber.destroy_node()
        #rclpy.shutdown()
    def close_tros(self):
        self.tros_activate = False
        self.label_img2.clear()
    
    
if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
