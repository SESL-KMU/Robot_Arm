import tensorflow as tf
from tensorflow.python.ops import gen_nn_ops
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.cm import get_cmap
import pickle

# 학습한 CNN모델을 LRP 알고리즘으로 입력의 데이터의 가중치 분석 코드

# LRP 관련 함수 및 클래스 github 참고하여 수정함, https://github.com/atulshanbhag/Layerwise-Relevance-Propagation
EPS = 1e-7

def get_model_params(model):
    names, activations, weights, strids = [], [], [], []

    for layer in model.layers:
        name = layer.name if layer.name != 'predictions' else 'fc_out'
        names.append(name)
        activations.append(layer.output)
        weights.append(layer.get_weights())
        if 'conv' in name: #conv 일때 stride 값 저장
            strids.append(layer.strides)#s[0])
        else:
            strids.append([])
    return names, activations, weights, strids

def gamma_correction(image, gamma=0.4, minamp=0, maxamp=None):
    c_image = np.zeros_like(image)
    image -= minamp
    if maxamp is None:
        maxamp = np.abs(image).max() + EPS
    image /= maxamp
    pos_mask = (image > 0)
    neg_mask = (image < 0)
    c_image[pos_mask] = np.power(image[pos_mask], gamma)
    c_image[neg_mask] = -np.power(-image[neg_mask], gamma)
    c_image = c_image * maxamp + minamp
    return c_image

def get_gammas(images, g=0.4, **kwargs):
    gammas = [gamma_correction(img, gamma=g, **kwargs) for img in images]
    return gammas

def project_image(image, output_range=(0, 1), absmax=None, input_is_positive_only=False):
    if absmax is None:
        absmax = np.max(np.abs(image), axis=tuple(range(1, len(image.shape))))
    absmax = np.asarray(absmax)
    mask = (absmax != 0)
    if mask.sum() > 0:
        image[mask] /= absmax[mask]
    if not input_is_positive_only:
        image = (image + 1) / 2
    image = image.clip(0, 1)
    projection = output_range[0] + image * (output_range[1] - output_range[0])
    return projection

def reduce_channels(image, axis=-1, op='sum'):
    if op == 'sum':
        return image.sum(axis=axis)
    elif op == 'mean':
        return image.mean(axis=axis)
    elif op == 'absmax':
        pos_max = image.max(axis=axis)
        neg_max = -((-image).max(axis=axis))
        return np.select([pos_max >= neg_max, pos_max < neg_max], [pos_max, neg_max])

def heatmap(image, cmap_type='rainbow', reduce_op='sum', reduce_axis=-1, **kwargs):
    cmap = get_cmap(cmap_type)
    shape = list(image.shape)
    reduced_image = reduce_channels(image, axis=reduce_axis, op=reduce_op)
    projected_image = project_image(reduced_image, **kwargs)
    heatmap = cmap(projected_image.flatten())[:, :3].T
    heatmap = heatmap.T
    shape[reduce_axis] = 3
    return heatmap.reshape(shape)

def get_heatmaps(gammas, cmap_type='rainbow', **kwargs):
    heatmaps = [heatmap(g, cmap_type=cmap_type, **kwargs) for g in gammas]
    return heatmaps

# LRP 클래스, 케라스 모델을 입력으로 함
class LayerwiseRelevancePropagation:

    def __init__(self, model_name, alpha=2, epsilon=1e-7):
        self.model = model_name
        self.alpha = alpha
        self.beta = 1 - alpha
        self.epsilon = epsilon

        self.names, self.activations, self.weights, self.strides = get_model_params(self.model)
        self.num_layers = len(self.names)

        self.relevance = self.compute_relevances()
        self.lrp_runner = tf.keras.backend.function(inputs=[self.model.input, ], outputs=[self.relevance, ])

    def compute_relevances(self):
        r = self.model.output
        for i in range(self.num_layers - 2, -1, -1):
            if 'dense' in self.names[i + 1]:
                r = self.backprop_fc(self.weights[i + 1][0], self.weights[i + 1][1], self.activations[i], r)
            elif 'flatten' in self.names[i + 1]:
                r = self.backprop_flatten(self.activations[i], r)
            elif 'pool' in self.names[i + 1]:
                r = self.backprop_max_pool2d(self.activations[i], r)
            elif 'conv2d' in self.names[i + 1]:
                r = self.backprop_conv2d(self.weights[i + 1][0], self.weights[i + 1][1], self.activations[i], r)
            elif 'conv1d' in self.names[i + 1]:
                r = self.backprop_conv1d(self.weights[i + 1][0], self.weights[i + 1][1], self.activations[i], r, strides=self.strides[i+1])
            elif 'dropout' in self.names[i + 1]:
                r = r
            else:
                raise 'Layer not recognized!'
                sys.exit()
        return r

    def backprop_fc(self, w, b, a, r):
        w_p = tf.maximum(w, 0.)
        b_p = tf.maximum(b, 0.)
        z_p = tf.matmul(a, w_p) + b_p + self.epsilon
        s_p = r / z_p
        c_p = tf.matmul(s_p, tf.transpose(w_p))

        w_n = tf.minimum(w, 0.)
        b_n = tf.minimum(b, 0.)
        z_n = tf.matmul(a, w_n) + b_n - self.epsilon
        s_n = r / z_n
        c_n = tf.matmul(s_n, tf.transpose(w_n))

        return a * c_p#(self.alpha * c_p + self.beta * c_n)

    def backprop_flatten(self, a, r):
        shape = a.get_shape().as_list()
        shape[0] = -1
        return tf.reshape(r, shape)

    def backprop_max_pool2d(self, a, r, ksize=(1, 2, 2, 1), strides=(1, 2, 2, 1)):
        z = tf.nn.pool2d(a, pool_size=ksize[1:-1], strides=strides[1:-1], padding='valid', pool_mode='max')

        z_p = tf.maximum(z, 0.) + self.epsilon
        s_p = r / z_p
        c_p = gen_nn_ops.max_pool_grad_v2(a, z_p, s_p, ksize, strides, padding='VALID')

        z_n = tf.minimum(z, 0.) - self.epsilon
        s_n = r / z_n
        c_n = gen_nn_ops.max_pool_grad_v2(a, z_n, s_n, ksize, strides, padding='VALID')

        return a * c_p #(self.alpha * c_p + self.beta * c_n)

    def backprop_conv2d(self, w, b, a, r, strides=(1, 1, 1, 1)):
        w_p = tf.maximum(w, 0.)
        b_p = tf.maximum(b, 0.)
        z_p = tf.nn.conv2d(a, filter=w_p, strides=strides, padding='SAME') + b_p + self.epsilon
        s_p = r / z_p
        c_p = tf.nn.conv2d_backprop_input(tf.shape(a), w_p, s_p, strides, padding='SAME')

        w_n = tf.minimum(w, 0.)
        b_n = tf.minimum(b, 0.)
        z_n = tf.nn.conv2d(a, filter=w_n, strides=strides, padding='SAME') + b_n - self.epsilon
        s_n = r / z_n
        c_n = tf.nn.conv2d_backprop_input(tf.shape(a), w_n, s_n, strides, padding='SAME')

        return a * c_p #(self.alpha * c_p + self.beta * c_n)

    def backprop_conv1d(self, w, b, a, r, strides=(1, 1, 1)):
        w_p = tf.maximum(w, 0.)
        b_p = tf.maximum(b, 0.)
        z_p = tf.nn.conv1d(a, filters=w_p, stride=strides, padding='VALID') + b_p + self.epsilon
        s_p = r / z_p
        in_shapt = [tf.shape(a)[0], 1, tf.shape(a)[1], tf.shape(a)[2]]
        if len(strides) != 3:
            in_strides = [1, 1, strides[0], 1]
        else:
            in_strides = [strides[0], 1, strides[1], strides[2]]
        c_p = tf.nn.conv2d_backprop_input(in_shapt, tf.expand_dims(w_p, axis=0), tf.expand_dims(s_p, axis=1), in_strides, padding='VALID')
        c_p = tf.reshape(c_p, shape=tf.shape(a))

        w_n = tf.minimum(w, 0.)
        b_n = tf.minimum(b, 0.)
        z_n = tf.nn.conv1d(a, filters=w_n, stride=strides, padding='VALID') + b_n - self.epsilon
        s_n = r / z_n
        c_n = tf.nn.conv2d_backprop_input(in_shapt, tf.expand_dims(w_n, axis=0), tf.expand_dims(s_n, axis=1), in_strides, padding='VALID')
        c_n = tf.reshape(c_n, shape=tf.shape(a))
        return a * c_p#(self.alpha * c_p + self.beta * c_n)

    # def predict_labels(self, images):
    #     return predict_labels(self.model, images)

    def run_lrp(self, data):
        # print("Running LRP on {0} data...".format(len(data)))
        return self.lrp_runner([data, ])[0]

    def compute_heatmaps(self, images, g=0.2, cmap_type='rainbow', **kwargs):
        lrps = self.run_lrp(images)
        print("LRP run successfully...")
        # gammas = get_gammas(lrps, g=g, **kwargs)
        # print("Gamma Correction completed...")
        # heatmaps = get_heatmaps(gammas, cmap_type=cmap_type, **kwargs)
        return lrps

if __name__ == '__main__':

    # 학습한 CNN 모델 로드
    model = tf.keras.models.load_model('trained_CNN.h5')

    # CNN 모델에 대한 LRP 객체 생성
    lrp = LayerwiseRelevancePropagation(model)

    # 가중치 판별을 할 학습 데이터 로드
    with open('../train_data.pickle', 'rb') as f:
        data = pickle.load(f)
        xtrain, ytrain = zip(*data)

    # CNN 모델 입력에 맞게 수정
    xtrain = np.reshape(xtrain, [-1, 6, 7680]).transpose([0, 2, 1])
    xtrain = np.array(xtrain)
    ytrain = np.array(ytrain)

    # 클래스별로 나누기
    c_data = [xtrain[np.where(ytrain == i)] for i in range(3)]

    # 각 클래스에 대해 LRP 수행
    lrp_classes = []
    for c in c_data:
        lrped = []
        for x in c:
            l = lrp.run_lrp(np.array([x]))[0]
            lrped.append(l)
        lrp_classes.append(lrped)

    # 각 클래스 LRP 결과들을 평균하여 각 클래스 별 가중치 맵을 저장
    # 값이 높을 수록 해당 분류를 하는데 중요한 부분으로 사용됨
    class_name = ['normal', 'ddock', 'grease_loss']

    for i, c in enumerate(c_data):
        c_ = np.mean(c, axis=0) # 클래스 주파수 평균
        c_ = np.transpose(c_)[:,::-1] # 주파수 변환할 때 처음과 끝이 반대로 변환 되었기에 다시 반전
        fig, axs = plt.subplots(6, 1, constrained_layout=True, dpi=300)
        for k, data in enumerate(c_): # 각 진동 축별로 그래프로 표시
            axs[k].plot(data, 'r')
        plt.savefig('lrp_result/mean_' +class_name[i] + '.png', dpi=300) # 그래프 저장

    for i, c in enumerate(lrp_classes):
        c_ = np.mean(c, axis=0) # 클래스 모든 LRP 결과를 평균
        c_ = np.transpose(c_)[:,::-1] # 주파수 변환할 때 처음과 끝이 반대로 변환 되었기에 다시 반전
        fig, axs = plt.subplots(6, 1, constrained_layout=True, dpi=300)
        for k, data in enumerate(c_): # 각 진동 축별로 그래프로 표시
            axs[k].plot(data, 'r')
        plt.savefig('lrp_result/lrp_' +class_name[i] + '.png', dpi=300) # 그래프 저장



