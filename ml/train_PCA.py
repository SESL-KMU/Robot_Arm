import numpy as np
import pickle
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier

# PCA와 KNN으로 모터 데이터 학습 코드

# train, test 데이터 로드
with open('../test_data.pickle', 'rb') as f:
    xtest, ytest = zip(*pickle.load(f))

with open('../train_data.pickle', 'rb') as f:
    xtrain, ytrain = zip(*pickle.load(f))

print('load data')

# scikit-learn 에서 제공하는 PCA 라이브러리 사용, 데이터의 95% 정보량을 남기고 데이터 축소
# 참고 : https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html
# PCA 모델 학습
pca = PCA(n_components=0.95)
X_reduced = pca.fit_transform(xtrain) # pca 학습과 동시에 train 데이터 변환(축소)
print(pca.n_components_) # 축소된 데이터 수
print(np.shape(X_reduced)) # 축소한 데이터 shape 확인 [samples, n_components_]

# KNN 학습, PCA는 분류 모델이 아니기에 분류가 가능한 KNN 사용
# 참고 : https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html
kVals = [3,5,7] # KNN의 K 값을 설정
accuracies = []

models = [] # 각 K 값에 따른 KNN 모델 리스트
# loop over kVals, K 값에 따라 KNN 모델 생성 및 학습
for k in kVals:
    # train the classifier with the current value of `k`
    model = KNeighborsClassifier(n_neighbors=k) # scikit-learn 에서 제공하는 KNN 라이브러리 사용
    model.fit(X_reduced, ytrain) # 축소된 데이터로 KNN 학습

    x_test_reduced = pca.transform(xtest) # PCA로 test 데이터 축소

    # evaluate the model and print the accuracies list
    score = model.score(x_test_reduced, ytest)
    print("k=%d, accuracy=%.2f%%" % (k, score * 100))
    accuracies.append(score)
    models.append(model)

best_k = np.argmax(accuracies) # 가장 정확도가 높은 모델 인덱스
print("k=%d achieved highest accuracy of %.2f%% on validation data" % (kVals[best_k],accuracies[best_k] * 100))

# PCA 모델 및 KNN 모델 저장
with open('trained_PCA.pickle', 'wb')as f:
    pickle.dump(pca, f)

with open('trained_KNN.pickle', 'wb')as f:
    pickle.dump(models[best_k], f)