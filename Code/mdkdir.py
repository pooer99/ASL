import numpy as np
import os

# 创建空文件夹

# 收集的手语数据的地址
DATA_PATH = os.path.join('../HP_Data')
# 将检测的手语
actions = np.array(['hello', 'thanks', 'iloveyou'])
# 训练视频数量
no_sequences = 30
# 视频帧数长度
sequence_length = 30

for action in actions:
    for sequence in range(no_sequences):
        try:
            # 创建文件夹，路径：DATA_PATH/action/sequence
            os.makedirs(os.path.join(DATA_PATH,action,str(sequence)))
        except:
            pass