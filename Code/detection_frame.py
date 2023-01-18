import cv2
import mediapipe as mp
import numpy as np

# mediapipe类，提供检测模型的相关方法

# mediapipe
mp_holistic = mp.solutions.holistic # 整体模型
mp_drawing = mp.solutions.drawing_utils # 绘图工具

# 使用模型检测图片
def mediapipe_detection(image, model):
    image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)# 颜色转换(模型检测需要是RGB的图片)
    image.flags.writeable = False # 设为不可写
    results = model.process(image)# 模型检测图片
    image.flags.writeable = True# 设为可写
    image = cv2.cvtColor(image,cv2.COLOR_RGB2BGR)# 颜色转换(还原图片)
    return image,results

# 渲染绘制骨骼结点
# 旧版
def draw_landmarks(image,results):
    mp_drawing.draw_landmarks(image, results.face_landmarks, mp_holistic.FACEMESH_CONTOURS)  # 绘制脸部结点
    mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS)  # 绘制姿势结点
    mp_drawing.draw_landmarks(image, results.left_hand_landmarks,mp_holistic.HAND_CONNECTIONS)  # 绘制左手结点
    mp_drawing.draw_landmarks(image, results.right_hand_landmarks,mp_holistic.HAND_CONNECTIONS)  # 绘制右手结点

# 新版，可更换风格:结点颜色、厚度、圆点半径
def draw_styled_landmarks(image,results):
    # 绘制脸部结点
    mp_drawing.draw_landmarks(image, results.face_landmarks, mp_holistic.FACEMESH_CONTOURS,
                              mp_drawing.DrawingSpec(color=(80,110,10), thickness=1, circle_radius=1),
                              mp_drawing.DrawingSpec(color=(80,256,121), thickness=1, circle_radius=1))
    # 绘制姿势结点
    mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS,
                              mp_drawing.DrawingSpec(color=(255,48,48), thickness=2, circle_radius=2),
                              mp_drawing.DrawingSpec(color=(48,48,255), thickness=2, circle_radius=2))
    # 绘制左手结点
    mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                              mp_drawing.DrawingSpec(color=(121,22,76), thickness=2, circle_radius=2),
                              mp_drawing.DrawingSpec(color=(80,256,121), thickness=2, circle_radius=2))
    # 绘制右手结点
    mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                              mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=2),
                              mp_drawing.DrawingSpec(color=(255,144,30), thickness=2, circle_radius=2))

# 提取关键结点
def extract_keypoints(results):
    pose = np.array([[res.x,res.y,res.z,res.visibility] for res in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(33*4)
    face = np.array([[res.x, res.y, res.z] for res in results.face_landmarks.landmark]).flatten() if results.face_landmarks else np.zeros(468*3)
    lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21*3)
    rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21*3)
    return np.concatenate([pose, face, lh, rh])