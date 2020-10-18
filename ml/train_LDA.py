import pickle
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn import metrics
import utils

# LDA로 모터 데이터 학습 코드

# train, test 데이터 로드
with open('../test_data.pickle', 'rb') as f:
    xtest, ytest = zip(*pickle.load(f))

with open('../train_data.pickle', 'rb') as f:
    xtrain, ytrain = zip(*pickle.load(f))

print('load data')

# scikit-learn 에서 제공하는 LDA 라이브러리 사용, class-1 개로 데이터 축소
# 참고 : https://scikit-learn.org/stable/modules/generated/sklearn.discriminant_analysis.LinearDiscriminantAnalysis.html
lda = LinearDiscriminantAnalysis(n_components=2)
print('start train')
lda.fit(xtrain, ytrain) # LDA 학습 시작

# 학습 후 test 데이터에 대하여 분류 및 정확도 측정
Y_predict2 = lda.predict(xtest)
score = metrics.accuracy_score(ytest, Y_predict2)
print('정확도는 ',score*100, '% 입니다')

# 학습한 LDA 모델 저장
with open('trained_LDA.pickle','wb') as f:
    pickle.dump(lda,f)


