import numpy as np
import pickle
import matplotlib.pyplot as plt
import pandas as pd

# 학습한 RF 모델로 정상에서 뚝이음 고장으로의 진행도 확인 코드
# RF는 예측시 확률로 출력 가능

# 학습한 모델 로드
with open('trained_RF.pickle','rb')as f:
    trained_RF = pickle.load(f)

# 각 일자별 데이터 로드
with open('../test_data_all_day_10.pickle', 'rb') as f:
    total_data, y = pickle.load(f)

print('load data')

# 각 일자별 데이터의 RF로 확률로 출력하여 표현
results = []
for i, data in enumerate(total_data):
    result = trained_RF.predict_proba(data)[0]
    results.append(result[:2]/sum(result[:2])) #정상과 뚝이음 확률만 표시
results = np.array(results)

# 각 일자변 정상과 뚝이음 확률을 그래프화

for i, y_ in enumerate(y):
    y[i] = y_.split('_')[0]

n_sample = 10
df = pd.DataFrame(results, index=y, columns=['normal', 'ddock'])
df.tail()
df = df.reset_index()
ax = df.plot(xticks=df.index[0::n_sample])
ax.set_xticklabels(y[0::n_sample], rotation=45)
plt.show()
