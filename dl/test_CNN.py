import numpy as np
import tensorflow as tf
import pickle
import matplotlib.pyplot as plt
import pandas as pd

# 학습한 CNN 모델로 정상에서 뚝이음 고장으로의 진행도 확인 코드

# 학습한 모델 로드
model_name = 'trained_CNN.h5'
model = tf.keras.models.load_model(model_name)

# 각 일자별 데이터 로드
with open('../test_data_all_day_10.pickle', 'rb') as f:
    total_data, y = pickle.load(f)
total_data = np.reshape(total_data, [-1, 6, 7680]).transpose([0, 2, 1]) #데이터를 CNN 입력 형태로 맞춤

print('load data')

# 각 일자별 데이터를 CNN으로 정상과 뚝이음 확률 예측
results = []
for i, data in enumerate(total_data):
    x = data
    result = model.predict(np.array([x]))[0]
    results.append(result[:2] / sum(result[:2]))  # 정상과 뚝이음 확률만 표시
results = np.array(results)

# 각 일자별 정상과 뚝이음 확률을 그래프화
for i, y_ in enumerate(y):
    y[i] = y_.split('_')[0]

n_sample = 10
df = pd.DataFrame(results, index=y, columns=['normal', 'ddock'])
df.tail()
df = df.reset_index()
ax = df.plot(xticks=df.index[0::n_sample])
ax.set_xticklabels(y[0::n_sample], rotation=45)
plt.show()

