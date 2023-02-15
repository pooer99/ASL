import os.path
import numpy as np
#使用from..import..时，只导入文件部分，文件不会自动运行；若使用import导入文件，则文件会自动运行
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import action_map as am

#用于直接调用model

actions = am.actions

#构建LSTM神经网络
model = Sequential()
model.add(LSTM(64, return_sequences=True, activation='relu', input_shape=(30,1662)))
model.add(LSTM(128, return_sequences=True, activation='relu'))
model.add(LSTM(64, return_sequences=False, activation='relu'))
model.add(Dense(64,activation='relu'))
model.add(Dense(32,activation='relu'))
model.add(Dense(actions.shape[0],activation='softmax'))

model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])

model.load_weights('../Model/test.h5')