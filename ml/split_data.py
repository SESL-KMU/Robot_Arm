import utils
import numpy as np
import pickle

# 각 클래스 별 모터 1, 2 데이터 경로
normal_motor1_path = 'H:/motor/2019/07/01/motor1/xyz/'
normal_motor2_path = 'H:/motor/2019/07/01/motor2/xyz/'
ddock_motor1_path = 'H:/motor/2019/08/02/motor1/xyz/'
ddock_motor2_path = 'H:/motor/2019/08/02/motor2/xyz/'
crack_motor1_path = 'H:/motor/2019/10/09/motor1/xyz/'
crack_motor2_path = 'H:/motor/2019/10/09/motor2/xyz/'

# 모터1과 모터2의 동시간에 측정된 데이터에 대해 경로를 합쳐줌
normal_motor = utils.make_dir_list(normal_motor1_path, normal_motor2_path)
ddock_motor = utils.make_dir_list(ddock_motor1_path, ddock_motor2_path)
crack_motor = utils.make_dir_list(crack_motor1_path, crack_motor2_path)

#각 클래스에 대하여 랜덤하게 셔플
np.random.shuffle(normal_motor)
np.random.shuffle(ddock_motor)
np.random.shuffle(crack_motor)
print(len(normal_motor), len(ddock_motor), len(crack_motor))

#각 클래스에 대해 80% 비율로 학습 데이터와 테스트 데이터로 분리
train1 = normal_motor[:int(len(normal_motor)*0.8)]
train2 = ddock_motor[:int(len(ddock_motor)*0.8)]
train3 = crack_motor[:int(len(crack_motor)*0.8)]

test1 = normal_motor[int(len(normal_motor)*0.8):]
test2 = ddock_motor[int(len(ddock_motor)*0.8):]
test3 = crack_motor[int(len(crack_motor)*0.8):]

#3개의 클래스 경로를 하나로 합쳐줌
train_list = [train1, train2, train3]
test_list = [test1, test2, test3]
print(len(train_list), len(test_list))

#train, test로 분류된 데이터 경로로 실제 데이터를 로드하여 저장함
xtrain, ytrain = utils.load_data(train_list)
xtest, ytest = utils.load_data(test_list)

with open('test_data.pickle', 'wb') as f:
    pickle.dump(list(zip(xtest, ytest)), f)

with open('train_data.pickle', 'wb') as f:
    pickle.dump(list(zip(xtrain, ytrain)), f)
