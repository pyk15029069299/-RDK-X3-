#!/usr/bin/env python3

import sys
import signal
import os
import numpy as np
import cv2
import colorsys
import dlib
from time import time
import math 

def shape_to_np(shape, dtype="int"):
    # initialize the list of (x, y)-coordinates
    coords = np.zeros((shape.num_parts, 2), dtype=dtype)

    # loop over all facial landmarks and convert them
    # to a 2-tuple of (x, y)-coordinates
    for i in range(0, shape.num_parts):
        coords[i] = (shape.part(i).x, shape.part(i).y)

    # return the list of (x, y)-coordinates
    return coords


def enhance_contrast(image):
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(image)
    return enhanced

def mouth_aspect_ratio(mouth):# 嘴部
    A = calculate_distance(mouth[2] , mouth[9])  # 51, 59
    B = calculate_distance(mouth[4] , mouth[7])  # 53, 57
    C = calculate_distance(mouth[0] , mouth[6])  # 49, 55
    mar = (A + B) / (2.0 * C)
    return mar

def calculate_distance(temp1, temp2):
    x1 = temp1[0]
    y1 = temp1[1]
    x2 = temp2[0]
    y2 = temp2[1]
    distance = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
    return distance


def eye_aspect_ratio(eye):
    # 垂直眼标志（X，Y）坐标
    A = calculate_distance(eye[1], eye[5])  # 计算两个集合之间的欧式距离
    B = calculate_distance(eye[2], eye[4])
    # 计算水平之间的欧几里得距离
    # 水平眼标志（X，Y）坐标
    C = calculate_distance(eye[0], eye[3])
    # 眼睛长宽比的计算
    ear = (A + B) / (2.0 * C)
    # 返回眼睛的长宽比
    return ear

