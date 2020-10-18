import numpy as np
import os
import struct
import csv
import pickle
import gzip
from scipy import signal

# 데이터 로드와 전처리 관련 함수 모음

def load_pickle(path): #모터 데이터 로드 함수
    with gzip.open(path, 'rb') as f:
        xyz_data = pickle.load(f)
    return xyz_data #np array shape = [32768*30, 3]

def xyzfft(xyz_data, down = 64): #진동 데이터를 주파수 데이터로 변환 및 64배 다운샘플링
    N = np.shape(xyz_data)[0]

    fft_data = []
    for j in range(np.shape(xyz_data)[1]): #모터데이터의 각 축(x,y,z)에 대하여 각각 퓨리에 변환 수행
        X = np.fft.fft(xyz_data[:, j])/N # numpy의 fast fourier transform 함수 사용
        X = np.fft.fftshift(X) #중심이 0이 되도록 shift
        X = abs(X) #magnitude 추출
        X = X[:len(X) // 2]*2 #32768*30/2
        X = signal.resample(X, len(X)//down) #32768*30/2/64 = 7680
        fft_data.append(X)
    # 3, 7680
    return np.array(fft_data).transpose() # [7680, 3]

def load_from_list(filelist, fft=True): #여러 경로에서 데이터 로드 함수
    data = []
    for i in filelist:
        m1, m2 = i
        if not '.pickle' in m1 or not '.pickle' in m2:
            continue

        xyz_data1 = load_pickle(m1)
        xyz_data2 = load_pickle(m2)

        # 데이터 중 측정안된 경우가 있기에 제외 하는 조건문
        minmax = np.max(xyz_data1) - np.min(xyz_data1)
        if minmax < 0.1:
            # print('m1 data not measured')
            continue
        minmax = np.max(xyz_data2) - np.min(xyz_data2)
        if minmax < 0.1:
            # print('m2 data not measured')
            continue
        # 30초 측정 안된 경우가 있기에 제외하는 조건문
        if np.shape(xyz_data1)[0] != 983040:
            continue
        #주파수 변환, 30초 데이터를 한번에 변환함
        if fft:
            a = xyzfft(xyz_data1) #모터1 데이터 변환
            b = xyzfft(xyz_data2) #모터2 데이터 변환
            data.append(np.concatenate([a, b], axis=1)) #모터1, 모터2 합침, [7680, 6]
        else: #주파수 변환 안할시 데이터 합침 [983040,6]
            data.append(np.concatenate([xyz_data1, xyz_data2], axis=1))
    return np.array(data)



def load_data(data_list): #train, test로 나누 데이터 로드하는 함수
    c1, c2, c3 = data_list # 클래스별 데이터로 나눔, 정상, 뚝이음, 크랙 순
    b1 = load_from_list(c1)  # 데이터를 로드, [sample, 32786*30/2/64, 6], sample은 각 클래스 데이터의 개수
    b2 = load_from_list(c2)
    b3 = load_from_list(c3)

    label0 = np.full([np.shape(b1)[0]], 0)  # 정상동작 labels값 0을 만들어줌
    label1 = np.full([np.shape(b2)[0]], 1)  # 뚝뚝동작 labels값 1을 만들어줌
    label2 = np.full([np.shape(b3)[0]], 2)  # 크랙동작 labels값 2를 만들어줌

    xdata = np.concatenate([b1, b2, b3], axis=0)  # 각 클래스 데이터를 합침 [sample(정상+뚝이음+크랙), 32786*30/2/64, 6]
    xdata = np.transpose(xdata, axes=[0, 2, 1])  # 데이터의 진동 축과 주파수 위치 수정 [sample(정상+뚝이음+크랙), 6, 32786*30/2/64]
    # PCA, LDA, RF 모델은 1차원 데이터만 입력으로 받으므로 [sample, feature] 형태로 reshape
    xdata = np.reshape(xdata, [-1, np.shape(xdata)[1] * np.shape(xdata)[2]])

    ydata = np.concatenate([label0, label1, label2], axis=0)  # 각 클래스 label 합침
    xdata = xdata.astype(np.float32) # 데이터 타입 변경

    return xdata, ydata

def make_dir_list(motor1_path, motor2_path): # 모터1과 모터2의 같은 시간대의 경로끼리 결합, 완전히 같은 시간에 측정되지 않고 1~2초 차가 있을 수 있음
    m1List = os.listdir(motor1_path)
    m2List = os.listdir(motor2_path)

    outList = []

    for i in range(len(m1List)):
        p1 = m1List[i]
        if not '.pickle' in p1:
            continue
        # print(p1.split('.')[0], p1.split('.')[0].split('_')[1])
        p1day = int(p1.split('.')[0].split('_')[0])
        p1time = int(p1.split('.')[0].split('_')[1])
        for j in range(len(m2List)):
            p2 = m2List[j]
            if not '.pickle' in p2:
                continue
            p2day = int(p2.split('.')[0].split('_')[0])
            p2time = int(p2.split('.')[0].split('_')[1])

            if p2day==p1day and abs(p1time-p2time) <= 2:
                # print(motor1_path+p1, motor2_path+p2)
                outList.append([motor1_path+p1, motor2_path+p2])
                m2List.pop(j)
                break
    return outList

