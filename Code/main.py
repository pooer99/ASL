# 1.导入依赖包
import detection_frame as df
import cv2
import numpy as np
from using_model import model
import action_map as am

# 检测变量
# 用于存储实时的帧，满30帧后，进行预测
sequence = []
sentence = []
# 阈值
threshold = 0.4

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

        # 预测
        keypoints = df.extract_keypoints(results)
        sequence.append(keypoints)
        sequence = sequence[-30:]

        if len(sequence) == 30:
            res = model.predict(np.expand_dims(sequence, axis=0))[0]
            print(am.actions[np.argmax(res)])

        # 显示在窗口中
        cv2.imshow('ASL', image)
        # 按q退出
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

# 释放
cap.release()
cv2.destroyAllWindows()

#2.04.06