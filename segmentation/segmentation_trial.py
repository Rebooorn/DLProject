from PIL import Image
import pickle
import time
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import glob
import os

# Load dataset for semantic segmentation
# location of the files
camvidpath = r'D:\\ChangLiu\\SegNet\\SegNet-Tutorial\\CamVid\\'
# training data
path1 = camvidpath + r'train\\'
path2 = camvidpath + r'trainannot\\'
trainimglist = glob.glob(path1+'*.png')
trainannotlist = glob.glob(path2+'*.png')
print(len(trainimglist), ' train images')
print(len(trainannotlist), ' train annotation')

# test data
path1 = camvidpath + r'test\\'
path2 = camvidpath + r'testannot\\'
testimglist = glob.glob(path1+'*.png')
testannotlist = glob.glob(path2+'*.png')
print(len(testimglist), ' test images')
print(len(testannotlist), ' test annotation')

# get train/test images
height = 128
width = height
nrclass = 22
trainData = None
trainLabel = None
trainLabelOnehot = None
trainlen = len(trainimglist)
testData = None
testLabel = None
testLabelOnehot = None
testlen = len(testimglist)


def DenseToOneHot(labels_dense, num_classes):
    '''this function transform labels into one-hot array'''
    n = labels_dense.shape[0] # num of labels
    labels_one_hot = np.zeros([n, num_classes])
    labels_one_hot[np.arange(n), labels_dense] = 1
    return labels_one_hot


''' train data process '''
for (f1, f2, i) in zip(trainimglist, trainannotlist, range(trainlen)):
    # train image
    img1 = Image.open(f1)
    img1 = img1.resize((height, width))
    rgb = np.array(img1).reshape(1, height, width, 3)
    # train model
    img2 = Image.open(f2)
    img2 = img2.resize((height, width))
    label = np.array(img2).reshape(1, height, width, 1)

    # stack images and labels
    if i == 0:
        # plt.imshow(label[0,:,:,0])
        # plt.show()
        trainData = rgb
        trainLabel = label
    else:
        trainData = np.concatenate((trainData, rgb), axis=0)
        trainLabel = np.concatenate((trainLabel, label), axis=0)

ntrain = len(trainData)

# One-hot label
trainLabelOnehot = np.zeros([trainLabel.shape[0], trainLabel.shape[1], trainLabel.shape[2], nrclass])
for row in range(height):
    for col in range(width):
        single = trainLabel[:, row, col, 0]
        oneHot = DenseToOneHot(single, nrclass)
        trainLabelOnehot[:, row, col, :] = oneHot
print('Train data process done')


'''Test data process'''
for (f1, f2, i) in zip(testimglist, testannotlist, range(testlen)):
    # test image
    img1 = Image.open(f1)
    img1 = img1.resize((height, width))
    rgb = np.array(img1).reshape(1, height, width, 3)
    # test model
    img2 = Image.open(f2)
    img2 = img2.resize((height, width))
    label = np.array(img2).reshape(1, height, width, 1)

    # stack images and labels
    if i == 0:
        # plt.imshow(label[0,:,:,0])
        # plt.show()
        testData = rgb
        testLabel = label
    else:
        testData = np.concatenate((testData, rgb), axis=0)
        testLabel = np.concatenate((testLabel, label), axis=0)

ntest = len(testData)

# One-hot label
testLabelOnehot = np.zeros([testLabel.shape[0], testLabel.shape[1], testLabel.shape[2], nrclass])
for row in range(height):
    for col in range(width):
        single = testLabel[:, row, col, 0]
        oneHot = DenseToOneHot(single, nrclass)
        testLabelOnehot[:, row, col, :] = oneHot
print('test data process done')

print("Shape of 'trainData' is %s" % (trainData.shape,))
print("Shape of 'trainLabel' is %s" % (trainLabel.shape,))
print("Shape of 'trainLabelOneHot' is %s" % (trainLabelOnehot.shape,))
print("Shape of 'testData' is %s" % (testData.shape,))
print("Shape of 'testLabel' is %s" % (testLabel.shape,))
print("Shape of 'testLabelOneHot' is %s" % (testLabelOnehot.shape,))

# Define Network
x = tf.placeholder(tf.float32, shape=[None, height, width, 3])
y = tf.placeholder(tf.float32, shape=[None, height, width, nrclass])
keepprob = tf.placeholder(tf.float32)
# kernels
ksize = 5
fsize = 64
initstdev = 0.01
initfun = tf.random_normal_initializer(mean=0.0, stddev=initstdev)
# initfun
weights = {
    'ce1': tf.get_variable('ce1', shape=[ksize, ksize, 3, fsize], initializer=initfun),
    'ce2': tf.get_variable('ce2', shape=[ksize, ksize, fsize, fsize], initializer=initfun),
    'ce3': tf.get_variable('ce3', shape=[ksize, ksize, fsize, fsize], initializer=initfun),
    'ce4': tf.get_variable('ce4', shape=[ksize, ksize, fsize, fsize], initializer=initfun),
    'cd4': tf.get_variable('cd4', shape=[ksize, ksize, fsize, fsize], initializer=initfun),
    'cd3': tf.get_variable('cd3', shape=[ksize, ksize, fsize, fsize], initializer=initfun),
    'cd2': tf.get_variable('cd2', shape=[ksize, ksize, fsize, fsize], initializer=initfun),
    'cd1': tf.get_variable('cd1', shape=[ksize, ksize, fsize, fsize], initializer=initfun),
    'dense_inner_prod': tf.get_variable('dense_inner_prod', shape=[1, 1, fsize, nrclass], initializer=initfun)
}
bias = {
    'be1': tf.get_variable('be1', shape=[fsize], initializer=tf.constant_initializer(value=0.0)),
    'be2': tf.get_variable('be2', shape=[fsize], initializer=tf.constant_initializer(value=0.0)),
    'be3': tf.get_variable('be3', shape=[fsize], initializer=tf.constant_initializer(value=0.0)),
    'be4': tf.get_variable('be4', shape=[fsize], initializer=tf.constant_initializer(value=0.0)),
    'bd4': tf.get_variable('bd4', shape=[fsize], initializer=tf.constant_initializer(value=0.0)),
    'bd3': tf.get_variable('bd3', shape=[fsize], initializer=tf.constant_initializer(value=0.0)),
    'bd2': tf.get_variable('bd2', shape=[fsize], initializer=tf.constant_initializer(value=0.0)),
    'bd1': tf.get_variable('bd1', shape=[fsize], initializer=tf.constant_initializer(value=0.0))
}

# DeconvNet model
def unpooling(inputOrg, size, mask = None):
    m = size[0]
    h = size[1]
    w = size[2]
    c = size[3]
    input = tf.transpose(inputOrg, [0, 3, 1, 2])
    x = tf.reshape(input, [-1, 1])
    k = np.float32(np.array([1.0, 1.0]).reshape([-1, 1]))
    output = tf.matmul(x, k)
    output = tf.reshape(output,[-1, c, h, w * 2])
    # m, c, w, h
    xx = tf.transpose(output, [0, 1, 3, 2])
    xx = tf.reshape(xx,[-1, 1])
    output = tf.matmul(xx, k)
    # m, c, w, h
    output = tf.reshape(output, [-1, c, w * 2, h * 2])
    output = tf.transpose(output, [0, 3, 2, 1])
    outshape = tf.stack([m, h * 2, w * 2, c])
    if mask != None:
        dense_mask = tf.sparse_to_dense(mask, outshape, output, 0)
        return output, dense_mask
    else:
        return output


def Model(_X, _W, _b, _keepprod):
    use_bias = 1
    #encoder 128*128
    encoder1 = tf.nn.conv2d(_X, _W['ce1'], strides=[1,1,1,1], padding='SAME')
    if use_bias is True:
        encoder1 = tf.nn.bias_add(encoder1, _b['be1'])
    mean, var = tf.nn.moments(encoder1, [0, 1, 2])
    encoder1 = tf.nn.batch_normalization(encoder1, mean, var, 0, 1, 0.0001)
    encoder1 = tf.nn.relu(encoder1)
    encoder1 = tf.nn.max_pool(encoder1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    encoder1 = tf.nn.dropout(encoder1, _keepprod)

    # 64*64
    encoder2 = tf.nn.conv2d(encoder1, _W['ce2'], strides=[1, 1, 1, 1], padding='SAME')
    if use_bias is True:
        encoder2 = tf.nn.bias_add(encoder2, _b['be2'])
    mean, var = tf.nn.moments(encoder2, [0, 1, 2])
    encoder2 = tf.nn.batch_normalization(encoder2, mean, var, 0, 1, 0.0001)
    encoder2 = tf.nn.relu(encoder2)
    encoder2 = tf.nn.max_pool(encoder2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    encoder2 = tf.nn.dropout(encoder2, _keepprod)

    # 32*32
    encoder3 = tf.nn.conv2d(encoder2, _W['ce3'], strides=[1, 1, 1, 1], padding='SAME')
    if use_bias is True:
        encoder3 = tf.nn.bias_add(encoder3, _b['be3'])
    mean, var = tf.nn.moments(encoder3, [0, 1, 2])
    encoder3 = tf.nn.batch_normalization(encoder3, mean, var, 0, 1, 0.0001)
    encoder3 = tf.nn.relu(encoder3)
    encoder3 = tf.nn.max_pool(encoder3, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')
    encoder3 = tf.nn.dropout(encoder3, _keepprod)

    # 16*16
    encoder4 = tf.nn.conv2d(encoder3, _W['ce4'], strides=[1, 1, 1, 1], padding='SAME')
    if use_bias is True:
        encoder4 = tf.nn.bias_add(encoder4, _b['be4'])
    mean, var = tf.nn.moments(encoder4, [0, 1, 2])
    encoder4 = tf.nn.batch_normalization(encoder4, mean, var, 0, 1, 0.0001)
    encoder4 = tf.nn.relu(encoder4)
    encoder4 = tf.nn.max_pool(encoder4, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    encoder4 = tf.nn.dropout(encoder4, _keepprod)

    # 8*8
    # Decoder 8*8 fsize: 64
    decoder4 = unpooling(encoder4, [tf.shape(_X)[0], height/16, width/16, fsize])
    decoder4 = tf.nn.conv2d_transpose(decoder4, _W['cd4'], tf.stack([tf.shape(_X)[0], ksize, ksize, fsize]), strides=[1,1,1,1], padding='SAME')
    if use_bias:
        decoder4 = tf.nn.bias_add(decoder4, _b['bd4'])
    mean, var = tf.nn.moments(decoder4, [0, 1, 2])
    decoder4 = tf.nn.batch_normalization(decoder4, mean, var, 0, 1, 0.0001)
    decoder4 = tf.nn.relu(decoder4)
    decoder4 = tf.nn.dropout(decoder4, _keepprod)

    # 16*16
    decoder3 = unpooling(decoder4, [tf.shape(_X)[0], height/8, height/8, fsize])
    decoder3 = tf.nn.conv2d(decoder3, _W['cd3'], strides=[1, 1, 1, 1], padding='SAME')
    if use_bias:
        decoder3 = tf.nn.bias_add(decoder3, _b['bd3'])
    mean, var = tf.nn.moments(decoder3, [0, 1, 2])
    decoder3 = tf.nn.batch_normalization(decoder3, mean, var, 0, 1, 0.0001)
    decoder3 = tf.nn.relu(decoder3)
    decoder3 = tf.nn.dropout(decoder3, _keepprod)

    # 32 * 32
    decoder2 = unpooling(decoder3, [tf.shape(_X)[0], height/4, height/4, fsize])
    decoder2 = tf.nn.conv2d(decoder2, _W['cd2'], strides=[1, 1, 1, 1], padding='SAME')
    if use_bias:
        decoder2 = tf.nn.bias_add(decoder2, _b['bd2'])
    mean, var = tf.nn.moments(decoder2, [0, 1, 2])
    decoder2 = tf.nn.batch_normalization(decoder2, mean, var, 0, 1, 0.0001)
    decoder2 = tf.nn.relu(decoder2)
    decoder2 = tf.nn.dropout(decoder2, _keepprod)

    # 64 * 64
    decoder1 = unpooling(decoder2, [tf.shape(_X)[0], height / 2, height / 2, fsize])
    decoder1 = tf.nn.conv2d(decoder1, _W['cd1'], strides=[1, 1, 1, 1], padding='SAME')
    if use_bias:
        decoder1 = tf.nn.bias_add(decoder1, _b['bd1'])
    mean, var = tf.nn.moments(decoder1, [0, 1, 2])
    decoder1 = tf.nn.batch_normalization(decoder1, mean, var, 0, 1, 0.0001)
    decoder1 = tf.nn.relu(decoder1)
    decoder1 = tf.nn.dropout(decoder1, _keepprod)

    # 128 * 128
    output = tf.nn.conv2d(decoder1, _W['dense_inner_prod'], strides=[1,1,1,1], padding='SAME')
    return output


# define function
pred = Model(x, weights, bias, keepprob)
lin_pred = tf.reshape(pred, shape=[-1, nrclass])
lin_y = tf.reshape(y, shape=[-1, nrclass])
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(lin_pred, lin_y))
# class label
predmax = tf.argmax(pred, axis=3)
ymax = tf.argmax(y, axis=3)
# Accuracy
corr = tf.equal(tf.argmax(y, 3), tf.argmax(pred, 3))
accr = tf.reduce_mean(tf.cast(corr, 'float'))
# Optimizer
optm = tf.train.AdamOptimizer(0.0001).minimize(cost)
batch_size = 128
n_epoch = 1000


















