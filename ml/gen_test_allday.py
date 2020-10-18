import numpy as np
import utils
import pickle
import os

# 모든 일자의 모터데이터 중 일자별로 n_sample 개의 데이터를 전처리하여 저장하는 코드

n_sample = 10

# 모든 일자의 모터데이터 경로
basepath = 'H:/motor/'

# 모든 일자의 데이터 중 각 일자별 10개의 데이터 경로 설정
baselist = os.listdir(basepath)
total_list = np.zeros([0, 2])
for year in baselist[:1]:
    folder_path=year+'/'
    month_list = os.listdir(basepath+folder_path)
    for month in month_list[1:3]:
        folder_path=year+'/'+month+'/'
        day_list = os.listdir(basepath+folder_path)
        for day in day_list:
            folder_path = year+'/'+month+'/'+ day + '/'
            m1_path = basepath+folder_path + 'motor1/xyz/'
            m2_path = basepath + folder_path + 'motor2/xyz/'
            if not os.path.isdir(m1_path) or not os.path.isdir(m2_path):
                print(m1_path)
                continue
            try:
                m_list = utils.make_dir_list(m1_path, m2_path)
                m_list = np.array(m_list)[np.arange(0, len(m_list), len(m_list) // n_sample)]
                total_list = np.concatenate([total_list, m_list], axis=0)
            except Exception as ex:
                print(m1_path, m2_path, ex)
                continue
path_list = total_list

# 생성한 데이터 경로로 데이터 전처리하여 로드
results = []
total_data = []
y_ = []
for i, p in enumerate(path_list):
    data = utils.load_from_list([p], fft=True)
    if len(data) == 0:
        continue
    if np.shape(data)[1] != 7680:
        continue
    data = np.transpose(data, axes=[0, 2, 1])
    data = np.reshape(data, [-1, np.shape(data)[1]*np.shape(data)[2]])
    total_data.append(data)
    y_.append(p)

# 각 데이터 경로로 부터 날짜 변환
y = []
for p in y_:
    y.append(p[0].split('.')[0].split('/')[-1])

# 전처리하여 로드한 데이터 저장
with open('../test_data_all_day_%d.pickle'%n_sample, 'wb') as f:
    pickle.dump([total_data, y], f)
