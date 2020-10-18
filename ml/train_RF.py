import pickle
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier

# Random Forest로 모터 데이터 학습 코드

# train, test 데이터 로드
with open('../test_data.pickle', 'rb') as f:
    xtest, ytest = zip(*pickle.load(f))

with open('../train_data.pickle', 'rb') as f:
    xtrain, ytrain = zip(*pickle.load(f))

print('load data')

# scikit-learn 에서 제공하는 Random Forest 라이브러리 사용
# 참고 : https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html
rf = RandomForestClassifier(n_estimators=10, n_jobs=4, random_state=2, warm_start=True)
rf.fit(xtrain, ytrain) # 학습

# test 데이터 분류 및 정확도 측정
y_=rf.predict(xtest)
score = metrics.accuracy_score(ytest, y_)
print('정확도는 ',score*100, '% 입니다')

# 학습 모델 저장
with open('trained_RF.pickle', 'wb')as f:
    pickle.dump(rf, f)

