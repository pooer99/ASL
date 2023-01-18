import detection_frame as df
import collection_keypoints as ck
import cv2
import numpy as np
import os
from matplotlib import pyplot as plt
import time

DATA_PATH = os.path.join('../HP_Data')
# 将检测的手语
actions = np.array(['hello','thanks','iloveyou'])
# 训练视频数量
no_sequences = 30
# 视频帧数长度
sequence_length = 30


cap = cv2.VideoCapture(0)
# 检测置信度0.5，跟踪置信度0.5
with df.mp_holistic.Holistic(min_detection_confidence=0.5,min_tracking_confidence=0.5) as holistic:

        for action in actions:
                for sequence in range(no_sequences):
                        for frame_num in range(sequence_length):
                                # ret为bool类型，指示是否成功读取这一帧
                                # 获取摄像头帧
                                ret, frame = cap.read()

                                # 对帧进行检测并返回检测结果results
                                image, results = df.mediapipe_detection(frame, holistic)

                                # 绘制骨骼结点
                                df.draw_styled_landmarks(image, results)

                                #开始收集
                                if frame_num == 0:
                                        cv2.putText(image,'STARTING COLLECTION',(120,200),
                                                cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),4,cv2.LINE_AA)
                                        cv2.putText(image, 'Collecting frams for {} video number {}'.format(action,sequence), (15, 12),
                                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 4, cv2.LINE_AA)
                                        cv2.waitKey(2000)
                                else:
                                        cv2.putText(image,'Collecting frams for {} video number {}'.format(action, sequence),(15, 12),
                                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 4, cv2.LINE_AA)

                                key_points = df.extract_keypoints(results)
                                npy_path = os.path.join(DATA_PATH,action,str(sequence),str(frame_num))
                                np.save(npy_path,key_points)

                                # 显示在窗口中
                                cv2.imshow('ASL', image)
                # 按q退出
                if cv2.waitKey(10) & 0xFF == ord('q'):
                    break

        # 释放
        cap.release()
        cv2.destroyAllWindows()
