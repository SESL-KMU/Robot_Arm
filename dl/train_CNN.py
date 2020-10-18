import numpy as np
import tensorflow as tf
import pickle
import matplotlib.pyplot as plt
import pandas as pd

# CNN 모델로 정상, 뚝이음, 크랙 분류 및 예측 학습 코드

# 학습 및 테스트 데이터 로드
with open('../test_data.pickle', 'rb') as f:
    xtest, ytest = zip(*pickle.load(f))

with open('../train_data.pickle', 'rb') as f:
    xtrain, ytrain = zip(*pickle.load(f))

# 데이터 shape 변경, CNN으로 학습할 것이기에 주파수와 진동 축을 분리하고 [sample, 7680, 6] 형태로 transpose
xtrain = np.reshape(xtrain, [-1, 6, 7680]).transpose([0,2,1])
xtest = np.reshape(xtest, [-1, 6, 7680]).transpose([0,2,1])
ytrain = np.expand_dims(ytrain, axis=-1)
ytest = np.expand_dims(ytest, axis=-1)

# 라벨 데이터 one hot 형태로 변환
ytrain = tf.keras.utils.to_categorical(np.array(ytrain))
ytest = tf.keras.utils.to_categorical(np.array(ytest))

# CNN 케라스 모델 선언
# 모터 데이터가 2차원 입력이기에 Conv1d를 사용하여 특징 추출 후 Dense 레이어로 분류
inputs = tf.keras.layers.Input(shape=[7680, 6])
x = tf.keras.layers.Conv1D(16, 100, 10, activation='relu')(inputs) # input_shape=[7680, 6])
x = tf.keras.layers.Dropout(0.3)(x)
x = tf.keras.layers.Conv1D(32, 10, 5, activation='relu')(x)
x = tf.keras.layers.Dropout(0.3)(x)
x = tf.keras.layers.Flatten()(x)
x = tf.keras.layers.Dense(3, activation='softmax')(x)

model = tf.keras.models.Model(inputs=inputs, outputs=x)

# 모델 형태 출력
model.summary()

# 모델 설정
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# 학습 파라미터 설정
batch_size = 256
epochs = 10

# 학습 시작, 학습 후 모델 저장하고 테스트 데이터 평가
model.fit(xtrain, ytrain, batch_size=batch_size, epochs=epochs, validation_data=[xtest, ytest], verbose=2)
model.save('trained_CNN.h5')
model.evaluate(xtest, ytest, batch_size=batch_size)