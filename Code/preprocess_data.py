import os.path
import numpy as np
#使用from..import..时，只导入文件部分，文件不会自动运行；若使用import导入文件，则文件会自动运行
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import TensorBoard
# 预测
from sklearn.metrics import multilabel_confusion_matrix, accuracy_score
import action_map as am

#----------训练数据----------

# 收集的手语数据的地址
DATA_PATH = os.path.join('../HP_Data')
# 将检测的手语
actions = am.actions
# 训练视频数量
no_sequences = 30
# 视频帧数长度
sequence_length = 30

#----------标签映射
#获取动作映射
label_map = {label:num for num,label in enumerate(am.actions)}

# 读取录制好的标签的动作npy数据
sequences, labels = [], [] # 存储文件路径，存储标签
for action in am.actions:
    for sequence in range(no_sequences):
        window = []
        for frame_num in range(sequence_length):
            # 获取动作数据资源
            res = np.load(os.path.join(DATA_PATH,action,str(sequence),"{}.npy".format(frame_num)))
            window.append(res)
        sequences.append(window)
        labels.append(label_map[action])

X = np.array(sequences)
# 初始化标签
y = to_categorical(labels).astype(int)

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.05)


#----------开始训练

#设置训练日志
log_dir = os.path.join('../Logs')
tb_callback = TensorBoard(log_dir=log_dir)

#构建LSTM神经网络
model = Sequential()
model.add(LSTM(64, return_sequences=True, activation='relu', input_shape=(30,1662)))
model.add(LSTM(128, return_sequences=True, activation='relu'))
model.add(LSTM(64, return_sequences=False, activation='relu'))
model.add(Dense(64,activation='relu'))
model.add(Dense(32,activation='relu'))
model.add(Dense(actions.shape[0],activation='softmax'))

model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])

# 查看神经网络结构
# model.summary()

'''
# 训练
model.fit(X_train, y_train, epochs=2000, callbacks=[tb_callback])

#存储训练模型
model.save('../Model/action.h5')

del model
'''

# 加载训练模型
model.load_weights('../Model/action.h5')

'''

# 使用混淆矩阵和准确性进行评估预测
yhat = model.predict(X_test)

ytrue = np.argmax(y_train, axis=1).tolist()
yhat = np.argmax(yhat, axis=1).tolist()

print(accuracy_score(ytrue, yhat))

'''

