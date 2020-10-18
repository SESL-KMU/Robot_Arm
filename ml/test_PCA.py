import numpy as np
import pickle
import matplotlib.pyplot as plt
import pandas as pd

# 학습한 PCA 모델로 정상에서 뚝이음 고장으로의 진행도 확인 코드

# 학습한 모델 로드
with open('trained_PCA.pickle','rb')as f:
    trainedPCA = pickle.load(f)

# 학습 데이터 로드
with open('../train_data.pickle', 'rb') as f:
    xtrain, ytrain = zip(*pickle.load(f))

# 각 일자별 데이터 로드
with open('../test_data_all_day_10.pickle', 'rb') as f:
    total_data, y = pickle.load(f)

print('load data')

# 학습데이터를 클래스별로 분리
xtrain = np.array(xtrain)
ytrain = np.array(ytrain)

c_data = [xtrain[np.where(ytrain==i)] for i in range(3)]
print(len(c_data), len(c_data[0]), len(c_data[1]), len(c_data[2]))

# 각 클래스별 PCA 변환 데이터의 평균 특징 구하기
mean_c_pca_data = []
c_pca_data = []
for c in c_data:
    pca_data = trainedPCA.transform(c)
    c_pca_data.append(np.array(pca_data))
    mean_c_pca_data.append(np.mean(pca_data, axis=0))
mean_c_pca_data = np.array(mean_c_pca_data)
print(np.shape(mean_c_pca_data))

#각 클래스별 분포 출력
plt.scatter(c_pca_data[0][:,0], c_pca_data[0][:,1], marker='o', label='normal')
plt.scatter(c_pca_data[1][:,0], c_pca_data[1][:,1], marker='^', label='ddock')
plt.scatter(c_pca_data[2][:,0], c_pca_data[2][:,1], marker='s', label='grease loss')
plt.legend()
plt.show()

# 각 일자별 데이터의 PCA 변환 특징과 클래스 평균 특징과 거리 비교, 확률로 출력
results = []
for i, data in enumerate(total_data):
    reduced = trainedPCA.transform(data)[0] # 다른 일자 데이터를 PCA 변환 (데이터 축소)
    # 정상과 뚝이음 평균 특징과 변환된 데이터의 거리, L2 Norm 사용
    result = np.linalg.norm(mean_c_pca_data - reduced, 2, axis=1)[:2]
    result = 1 - result / np.sum(result) # 거리를 확률로 표현
    results.append(result)
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

