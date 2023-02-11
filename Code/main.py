# 1.导入依赖包
import detection_frame as df
import collection_keypoints as ck
import cv2
import numpy as np
import os
from matplotlib import pyplot as plt
import time

# OpenCV连接摄像头
# 参数0-打开笔记本的内置摄像头
cap = cv2.VideoCapture(0)
# 检测置信度0.5，跟踪置信度0.5
with df.mp_holistic.Holistic(min_detection_confidence=0.5,min_tracking_confidence=0.5) as holistic:
    while cap.isOpened():
        # ret为bool类型，指示是否成功读取这一帧
        # 获取摄像头帧
        ret, frame = cap.read()

        # 对帧进行检测并返回检测结果results
        image, results = df.mediapipe_detection(frame, holistic)

        #绘制骨骼结点
        df.draw_styled_landmarks(image,results)

        # test

        # 显示在窗口中
        cv2.imshow('ASL', image)
        # 按q退出
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

# 释放
cap.release()
cv2.destroyAllWindows()

#1.43.28