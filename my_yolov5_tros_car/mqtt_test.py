from PyQt5.QtWidgets import QApplication, QMainWindow, QTextEdit, QLabel
from PyQt5 import QtGui
from PyQt5.uic import loadUi
from PyQt5.QtCore import Qt, QTimer, pyqtSignal
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
from mqtt.mqtt import MetricsTracker, MqttReporter, DISPLAY_FIELD_MAP  # 导入MQTT模块


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
        self.text_edit.setReadOnly(True)
        self.text_edit.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        
        self.label_img = QLabel(self.groupBox_1)
        self.label_img.setGeometry(10, 30, 400, 300)
        self.label_img.setAlignment(Qt.AlignTop | Qt.AlignLeft)
        
        self.pushButton_1.clicked.connect(self.open_camera1)
        self.pushButton_2.clicked.connect(self.shut_camera1)
        
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_camera)
        self.timer.start(1000 // 30) 
        
        self.PREDICTOR_PATH = "shape_predictor_68_face_landmarks.dat"
        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor(self.PREDICTOR_PATH)
        
        self.engine = pyttsx3.init()
        self.count_ear = 0
        self.count_mar = 0
        self.frame_counter = 0
        self.check_interval = 25
        self.is_blinking = False
        self.is_yawning = False
        
        self.pushButton_3.clicked.connect(self.open_tros)
        self.label_img2 = QLabel(self.groupBox_2)
        self.label_img2.setGeometry(10, 30, 400, 300)
        self.label_img2.setAlignment(Qt.AlignTop | Qt.AlignLeft)    
        self.pushButton_4.clicked.connect(self.close_tros)
        self.tros_activate = False

        rclpy.init()
        self.image_subscriber = ImageSubscriber()
        
        # ========== MQTT初始化 ==========
        self.metrics_tracker = MetricsTracker(
            closed_eye_timeout_seconds=2.0,
            behavior_confirm_frames=4,
            behavior_miss_tolerance=2
        )
        self.mqtt_reporter = MqttReporter(upload_interval=5.0)  # 每5秒上传一次
        
        # 添加定时器用于MQTT数据上传
        self.mqtt_timer = QTimer(self)
        self.mqtt_timer.timeout.connect(self.upload_mqtt_data)
        self.mqtt_timer.start(5000)  # 每5秒触发一次
        
        # 添加疲劳标志，防止同一事件重复计数
        self.fatigue_recorded = False
        
        # self.log_message("系统启动完成，MQTT已连接")

    def open_camera1(self):
        self.cam = srcampy.Camera()
        self.cam.open_cam(0, -1, 1, 672, 672)
        # self.log_message("摄像头1已打开")
    
    def update_camera(self):      
        if hasattr(self, 'cam'):
            img = self.cam.get_img(2, 672, 672)
            if img is not None:
                frame = np.frombuffer(img, dtype=np.uint8)
                frame = frame.reshape(672 * 3 // 2, 672)
                img_bgr = cv2.cvtColor(frame, cv2.COLOR_YUV2BGR_NV12)
                resized_img_bgr = cv2.resize(img_bgr, (320, 240), interpolation=cv2.INTER_AREA)
                
                gray_image = cv2.cvtColor(resized_img_bgr, cv2.COLOR_BGR2GRAY)
                rects = self.detector(gray_image)

                for face in rects:
                    landmarks = self.predictor(gray_image, face)
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
                    
                    # 检测眨眼
                    if ear < 0.2:
                        self.is_blinking = True
                        self.count_ear += 1
                    
                    # 检测打哈欠
                    if mar > 0.9:
                        self.is_yawning = True 
                        self.count_mar += 1
                        
                    self.frame_counter += 1
                    
                    if self.frame_counter >= self.check_interval:
                        self.frame_counter = 0
                        
                        # 疲劳检测逻辑
                        if self.count_ear > 1 or self.count_mar > 5:
                            # self.log_message(f"疲劳检测: 眨眼{self.count_ear}次, 打哈欠{self.count_mar}次")
                            
                            # 记录疲劳（防止重复计数）
                            if not self.fatigue_recorded:
                                self.metrics_tracker.record_fatigue()
                                self.fatigue_recorded = True
                                # self.log_message("⚠️ 疲劳事件已记录到物联网平台")
                            
                            # 语音提醒
                            volume = self.engine.getProperty('volume')
                            rate = self.engine.getProperty('rate')
                            self.engine.setProperty('volume', 10)
                            self.engine.setProperty('rate', rate - 100)
                            self.engine.say("Please rest")
                            self.engine.runAndWait()
                        else:
                            # 恢复正常，重置疲劳标志
                            self.fatigue_recorded = False
                        
                        # 重置计数器
                        self.count_ear = 0
                        self.count_mar = 0
                        self.is_blinking = False
                        self.is_yawning = False
                
                img_rgb = cv2.cvtColor(resized_img_bgr, cv2.COLOR_BGR2RGB)
                h, w, ch = img_rgb.shape
                bytes_per_line = ch * w
                q_img = QtGui.QImage(img_rgb.data, w, h, bytes_per_line, QtGui.QImage.Format_RGB888)
                pixmap = QtGui.QPixmap.fromImage(q_img)
                self.label_img.setPixmap(pixmap)
    
    def upload_mqtt_data(self):
        """定时上传MQTT数据"""
        # try:
            # 获取当前指标
        mqtt_metrics = self.metrics_tracker.build_mqtt_metrics()
        display_metrics = self.metrics_tracker.build_metrics()
        
        # 发布到MQTT
        success = self.mqtt_reporter.publish_if_due(mqtt_metrics)
            
            # if success:
            #     self.log_message(f"📤 数据已上传: 疲劳{display_metrics[DISPLAY_FIELD_MAP['fatigue_count']]}次 | "
            #                    f"驾驶时长{display_metrics[DISPLAY_FIELD_MAP['driving_duration_sec']]}秒 | "
            #                    f"安全评分{display_metrics[DISPLAY_FIELD_MAP['safety_score']]}")
        # except Exception as e:
            # self.log_message(f"❌ MQTT上传失败: {e}")
    
    # def log_message(self, message):
    #     """添加日志到文本编辑框"""
    #     current_text = self.text_edit.toPlainText()
    #     timestamp = time.strftime("%H:%M:%S")
    #     new_text = f"[{timestamp}] {message}\n"
    #     self.text_edit.setPlainText(current_text + new_text)
    #     # 自动滚动到底部
    #     scrollbar = self.text_edit.verticalScrollBar()
    #     scrollbar.setValue(scrollbar.maximum())
    
    def shut_camera1(self):
        if hasattr(self, 'cam'):
            self.cam.close_cam()
            del self.cam
            self.label_img.clear()
            self.log_message("摄像头1已关闭")

    def open_tros(self):
        image_subscriber = self.image_subscriber
        self.tros_activate = True
        # self.log_message("TROS系统已打开")
        
        while self.tros_activate:
            rclpy.spin_once(image_subscriber)
            image = image_subscriber.get_image()
            msg = image_subscriber.get_msg()  
            # if msg[1]:
                # self.log_message(f"检测: fps:{msg[0]} class:{msg[1]} conf:{round(msg[2],2)}")
            if image is not None:
                cv2.imshow('my_image', image)
                image = cv2.resize(image, (320, 240))
                image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                h, w, ch = image_rgb.shape
                bytes_per_line = ch * w
                q_image = QtGui.QImage(image_rgb.data, w, h, bytes_per_line, QtGui.QImage.Format_RGB888)
                pixmap = QtGui.QPixmap.fromImage(q_image)
                self.label_img2.setPixmap(pixmap)
                QApplication.processEvents()

    def close_tros(self):
        self.tros_activate = False
        self.label_img2.clear()
        # self.log_message("TROS系统已关闭")
    
    def closeEvent(self, event):
        """窗口关闭时的清理工作"""
        # self.log_message("系统关闭中...")
        # 关闭MQTT连接
        if hasattr(self, 'mqtt_reporter'):
            self.mqtt_reporter.close()
        # 停止定时器
        if hasattr(self, 'mqtt_timer'):
            self.mqtt_timer.stop()
        # 关闭摄像头
        self.shut_camera1()
        self.close_tros()
        event.accept()


# 需要添加的辅助函数（如果原来的shape_to_np等函数不在文件中）
def shape_to_np(shape, dtype="int"):
    """将dlib的shape对象转换为numpy数组"""
    coords = np.zeros((shape.num_parts, 2), dtype=dtype)
    for i in range(0, shape.num_parts):
        coords[i] = (shape.part(i).x, shape.part(i).y)
    return coords

def eye_aspect_ratio(eye):
    """计算眼睛纵横比"""
    A = np.linalg.norm(eye[1] - eye[5])
    B = np.linalg.norm(eye[2] - eye[4])
    C = np.linalg.norm(eye[0] - eye[3])
    ear = (A + B) / (2.0 * C)
    return ear

def mouth_aspect_ratio(mouth):
    """计算嘴巴纵横比"""
    A = np.linalg.norm(mouth[2] - mouth[10])
    B = np.linalg.norm(mouth[4] - mouth[8])
    C = np.linalg.norm(mouth[0] - mouth[6])
    mar = (A + B) / (2.0 * C)
    return mar


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())