import tensorflow as tf
import numpy as np
import os
import datetime
from PIL import Image
import matplotlib.pyplot as plt
from tensorflow.python.platform import gfile
from tensorflow.contrib import rnn


# 从 tfrecord 文件中解析结构化数据 （特征）
def parse_image_example(serialized_example):
    features = tf.parse_single_example(
        serialized_example,
        features={
            'image_raw': tf.FixedLenFeature([], tf.string),  # image data
            'pixels': tf.FixedLenFeature([], tf.int64),
            'label': tf.FixedLenFeature([], tf.int64),  # number label
        })
    return features


# 训练和测试数据
class DataBatch:
    def __init__(self, dir_train, dir_test, image_size, label_size):
        self.dir_train = dir_train
        self.dir_test = dir_test
        self.image_batch = None
        self.label_batch = None
        self.image_batch_test = None
        self.label_batch_test = None
        self.image_size = image_size
        self.label_size = label_size
        self.image_pixels = image_size[0]*image_size[1]

    # 准备训练和测试数据
    def prepare_data(self, batch_size, capacity, test_data_size, test_capacity):
        reader = tf.TFRecordReader()  # reader for TFRecord file
        if os.path.exists(self.dir_train) and os.path.exists(self.dir_test):
            train_queue = tf.train.string_input_producer([self.dir_train])
            test_queue = tf.train.string_input_producer([self.dir_test])  # files read queue
        else:
            raise Exception("%s or %s file doesn't exist" % (self.dir_train, self.dir_test))
        _, serialized_example = reader.read(train_queue)  # examples in TFRecord file
        _, serialized_test = reader.read(test_queue)
        features = parse_image_example(serialized_example)  # parsing features
        features_test = parse_image_example(serialized_test)
        pixels = tf.cast(features['pixels'], tf.int32)
        image = tf.decode_raw(features['image_raw'], tf.uint8)  # decode image data from string to image, a Tensor
        image.set_shape([self.image_pixels])  # pixels is 784
        label = tf.cast(features['label'], tf.int32)
        image_test = tf.decode_raw(features_test['image_raw'], tf.uint8)
        image_test.set_shape([self.image_pixels])
        label_test = tf.cast(features_test['label'], tf.int32)
        # self.image_batch, lb = tf.train.shuffle_batch(
        #     [image, label], batch_size=batch_size, capacity=capacity,
        #     min_after_dequeue=500)  # queue of image_batch, shuffle_batch mean random
        self.image_batch, lb = tf.train.batch(
            [image, label], batch_size=batch_size, capacity=capacity)  # queue of image_batch, shuffle_batch mean random
        self.label_batch = tf.one_hot(lb, self.label_size)  # one_hot, 2 for [0,0,1,0,0,0,...]
        self.image_batch_test, lb_test = tf.train.shuffle_batch(
            [image_test, label_test], batch_size=test_data_size, capacity=test_capacity, min_after_dequeue=0)
        self.label_batch_test = tf.one_hot(lb_test, self.label_size)


def input_num(input_str):
    while 1:
        a = input(input_str)
        if a == "":
            return a
        elif a.replace('.', '', 1).isdigit():
            return a
        else:
            print("wrong input!")


# 超参数 类
class Params:
    def __init__(self):
        self.batch_size = 128  # batch size of train data
        self.test_size = 300  # test data size 一次测试所用的数据量
        self.learning_rate = 0.001
        self.max_epochs = 10000
        self.expect_accuracy = 0.99
        self.n_layer = 3
        self.neurons = []
        self.output_dimension = 625
        self.patch_size = [3, 3]  # 卷积层的卷积核大小
        self.model_type = 'cnn'
        self.rnn_hidden_units = 128   # rnn hidden layer units

    # 输入超参数
    def input_params(self):
        while 1:
            cr = input("please choose CNN or RNN (cnn/rnn, default cnn): ")
            if cr != '':
                if cr != 'rnn' and cr != 'cnn':
                    print("wrong input! ")
                else:
                    self.model_type = cr
                    break
            else:
                break
        if self.model_type == 'cnn':
            f = input_num("please enter the number of convolution layer (3): ")
            if f != "":
                self.n_layer = int(f)
            for i in range(self.n_layer):
                g = input_num("%d convolution layer's neurons(%d): " % (i+1, 32*(2**i)))
                if g != "":
                    self.neurons.append(int(g))
                else:
                    self.neurons.append(32*(2**i))
            h = input_num("please enter the dimension of full connect layer(625): ")
            if h != "":
                self.output_dimension = int(h)
        elif self.model_type == 'rnn':
            k = input_num("please enter the number of rnn hidden layer units(128): ")
            if k != "":
                self.rnn_hidden_units = int(k)
        a = input_num("please enter the  batch size of train data (128): ")
        if a != "":
            self.batch_size = int(a)
        b = input_num("please enter the size of test data (300): ")
        if b != "":
            self.test_size = int(b)
        c = input_num("please enter the learning rate of optimizer (0.001): ")
        if c != "":
            self.learning_rate = float(c)
        d = input_num("please enter the maximal epochs of model training (10000): ")
        if d != "":
            self.max_epochs = int(d)
        e = input_num("please enter the expect accuracy of model (0.99): ")
        if e != "":
            self.expect_accuracy = float(e)


# 训练模型
class TrainModel:
    def __init__(self, ckpt_dir, graph_dir, log_dir):
        self.ckpt_dir = ckpt_dir  # 模型保存地址
        self.graph_dir = graph_dir  # graph地址
        self.log_dir = log_dir  # tensorboard log 地址

    # 训练的方法
    def model_training(self, model_params, input_data):

        X = tf.placeholder("float", [None, input_data.image_size[0], input_data.image_size[1], 1],
                           name="X")  # placeholder for model input(image data)
        Y = tf.placeholder("float", [None, input_data.label_size],
                           name="Y")  # placeholder for model output (prediction)

        if not os.path.exists(self.ckpt_dir):
            os.makedirs(self.ckpt_dir)

        p_keep_conv = tf.placeholder("float", name="p_keep_conv")
        p_keep_hidden = tf.placeholder("float", name="p_keep_hidden")
        if model_params.model_type == "cnn":
            py_x = model(X, model_params.n_layer, model_params.neurons, model_params.output_dimension,
                         p_keep_conv, p_keep_hidden, input_data.label_size, model_params.patch_size)  # 构建网络模型, 输出为py_x
        elif model_params.model_type == "rnn":
            py_x = rnn_model(X, input_data.image_size[0], input_data.image_size[1], model_params.rnn_hidden_units, 10)

        print("graph constructed success! ")
        saver = tf.train.Saver()  # saver for model

        cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=py_x, labels=Y))  # 代价函数

        train_op = tf.train.RMSPropOptimizer(model_params.learning_rate, 0.9).minimize(cost)  # 训练操作 使用RMS优化器

        train_op_adam = tf.train.AdamOptimizer(model_params.learning_rate).minimize(cost)

        predict_op = tf.argmax(py_x, 1, name="predict_op")

        start_time = datetime.datetime.now()

        with tf.Session() as sess:
            tf.global_variables_initializer().run()
            ckpt = tf.train.get_checkpoint_state(self.ckpt_dir)
            if ckpt and ckpt.model_checkpoint_path:
                print(ckpt.model_checkpoint_path)
            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(sess=sess, coord=coord)  # 启动队列
            accuracy = 0
            epochs = model_params.max_epochs
            print("start training ")
            for i in range(model_params.max_epochs):
                '''训练
                每次训练取出一个批次的数据，喂给train_op
                为了动态展示训练进程， 每次训练都打印出模型预测准确度
                '''
                bx, batch_ys = sess.run([input_data.image_batch, input_data.label_batch])
                batch_xs = bx.reshape(-1, input_data.image_size[0], input_data.image_size[1], 1)

                tex, tey = sess.run([input_data.image_batch_test, input_data.label_batch_test])
                tex = tex.reshape(-1, input_data.image_size[0], input_data.image_size[1], 1)
                if model_params.model_type == "cnn":
                    sess.run(train_op, feed_dict={X: batch_xs, Y: batch_ys, p_keep_conv: 0.8, p_keep_hidden: 0.5})
                    accuracy = np.mean(np.argmax(tey, axis=1) ==
                                       sess.run(predict_op, feed_dict={X: tex, p_keep_conv: 1.0, p_keep_hidden: 1.0}))
                elif model_params.model_type == "rnn":
                    sess.run(train_op_adam, feed_dict={X: batch_xs, Y: batch_ys})
                    accuracy = np.mean(np.argmax(tey, axis=1) ==
                                       sess.run(predict_op, feed_dict={X: tex}))
                print("The accuracy after %d train epochs is %.4f" % (i, accuracy))
                if accuracy >= model_params.expect_accuracy:
                    epochs = i + 1
                    break
            print("training completed ")
            writer = tf.summary.FileWriter(self.log_dir, tf.get_default_graph())
            writer.close()
            print("write tensorboard log success")
            tf.train.write_graph(sess.graph_def, self.graph_dir, "train.pb", False)
            print("save graph success")
            saver.save(sess, self.ckpt_dir + "/model.ckpt")
            print("save model success")
            coord.request_stop()
            coord.join(threads)
            end_time = datetime.datetime.now()
            duration = end_time - start_time
            print("accuracy: %2.2f%% training duration: %2d seconds  epochs: %2d "
                  % (accuracy * 100, duration.seconds, epochs))


# 初始化化权重
def init_weights(shape):
    return tf.Variable(tf.random_normal(shape, stddev=0.01))


# 神经网络结构
def model(X, n_layer, neurons, output_dimension, p_keep_conv, p_keep_hidden, label_size, patch_size):
    layer_input = 1
    lb = X
    image_min_wid_hei = min(X.get_shape().as_list()[1:2])  # image 的 宽和高的较小值
    for i in range(n_layer):
        layer_output = neurons[i]
        w = init_weights([patch_size[0], patch_size[1], layer_input, layer_output])
        la = tf.nn.relu(tf.nn.conv2d(lb, w, strides=[1, 1, 1, 1], padding='SAME'))  # 卷积层

        if round(image_min_wid_hei/(2**(i+1))) >= 4:
            l1 = tf.nn.max_pool(la, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
        else:
            l1 = la
        if i == n_layer - 1:
            wide = round(image_min_wid_hei/(2**n_layer))  #
            if wide < 4:
                wide = 4  # 池化(降采样)后，图片的最小宽度为4
            print("wide ", wide)
            w2 = init_weights([layer_output*wide*wide, output_dimension])
            l2 = tf.reshape(l1, [-1, w2.get_shape().as_list()[0]])
            print(l2)
        else:
            l2 = tf.nn.dropout(l1, p_keep_conv)

        layer_input = layer_output
        lb = l2
    w_o = init_weights([output_dimension, label_size])
    lo = tf.nn.relu(tf.matmul(lb, w2))
    lo = tf.nn.dropout(lo, p_keep_hidden)
    pyx = tf.matmul(lo, w_o)
    return pyx


def rnn_model(x, n_inputs, n_steps, n_hidden_units, n_classes):

    weights = {
        'out': tf.Variable(tf.random_normal([n_hidden_units, n_classes]))}
    biases = {
        'out': tf.Variable(tf.constant(0.1, shape=[n_classes, ]))}

    x = tf.reshape(x, [-1, n_steps, n_inputs])
    x = tf.unstack(x, n_steps, 1)

    lstm_cell = rnn.BasicLSTMCell(n_hidden_units, forget_bias=1.0)
    outputs, states = rnn.static_rnn(lstm_cell, x, dtype=tf.float32)
    results = tf.matmul(outputs[-1], weights['out']) + biases['out']

    return results


# 识别图片
def recognize_image(ckpt_dir, dir_be, graph_dir, image_size):
    # 手写图片预处理
    dir_af = "pic_for_predict"
    files1 = os.listdir(dir_af)
    for f in files1:
        os.remove(dir_af + "/" + f)
    if not os.path.exists(dir_be):
        raise Exception("%s dir doesn't exist " % dir_be)
    files = os.listdir(dir_be)
    cnt = len(files)
    if cnt == 0:
        raise Exception("there are no file")
    for i in range(cnt):
        img = Image.open(dir_be + "/" + files[i])
        img = img.resize((image_size[0], image_size[1]))
        img = img.convert("L")
        img.save(dir_af + "/" + files[i], "PNG")

    with tf.Session() as sess:
        ''' 加载保存好的图
        如果不进行这一步，也可以按照训练模型时一样的定义需要的张量
        '''
        with gfile.FastGFile(graph_dir+"/train.pb", "rb") as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())
            sess.graph.as_default()
            tf.import_graph_def(graph_def, name='')
        predict_op = sess.graph.get_tensor_by_name("predict_op:0")  # 取出预测值的张量
        X = sess.graph.get_tensor_by_name("X:0")  # 取出输入数据的张量
        p_keep_conv = sess.graph.get_tensor_by_name("p_keep_conv:0")  #
        p_keep_hidden = sess.graph.get_tensor_by_name("p_keep_hidden:0")  #

        # 加载保存好的模型
        saver = tf.train.import_meta_graph(ckpt_dir + "/model.ckpt.meta")

        ckpt = tf.train.get_checkpoint_state(ckpt_dir)
        if ckpt and ckpt.model_checkpoint_path:
            print(ckpt.model_checkpoint_path)
            saver.restore(sess, ckpt.model_checkpoint_path)

        # 开始识别图片
        files = os.listdir(dir_af)
        cnt = len(files)
        correct = 0
        for i in range(cnt):
            actual_label = int(files[i][0])
            files[i] = dir_af + "/" + files[i]
            img = Image.open(files[i])  # 读取要识别的图片
            print("input: ", files[i])
            imga = np.array(img).reshape(-1, image_size[0], image_size[1], 1)
            # feed 数据给 张量predict_op
            prediction = predict_op.eval(feed_dict={X: imga, p_keep_conv: 1.0, p_keep_hidden: 1.0})
            # 输出
            print("output: ", prediction)
            if prediction == actual_label:
                print("Correct!")
                correct = correct + 1
            else:
                print("Wrong!")
            print("\n")
        print("recognize finished")
        print("Verification accuracy is %.2f" % (correct / cnt))


def face_process(img):
    sigma = np.std(img)
    mean = np.mean(img)
    imgb = np.round((img-mean)/sigma*255)
    return imgb


if __name__ == "__main__":
    ckpt_dir = "./ckpt_dir_cnn"  # check point
    verify_pic_dir = "./test_num"  # 识别验证的文件路径  for MNIST data sets
    # verify_pic_dir = "./images"  # 识别验证的文件路径  for Fashion MNIST data sets
    graph_dir = "./graph"
    train_data_dir = "data/train.tfrecords"  # MNIST data sets
    # train_data_dir = "data/train.tfrecords"  # MNIST data sets
    # train_data_dir = "data/fashion_train.tfrecords"  # Fashion data sets

    test_data_dir = "data/test.tfrecords"
    # test_data_dir = "data/fashion_test.tfrecords"

    log_dir = "./logpath"  # path of tensorboard log
    capacity_of_train = 10000  # 训练数据队列的容量 , batch 函数参数
    capacity_of_test = 10000
    image_size = [28, 28]  # size of image in train data
    label_size = 10
    while 1:
        pattern = input("Please choose training pattern or recognizing(t/r): ")
        if pattern == "r" or pattern == "R":
            recognize_image(ckpt_dir, verify_pic_dir, graph_dir, image_size)
            break
        elif pattern == "t" or pattern == "T":
            params = Params()  # 初始化超参数
            params.input_params()  # 输入超参数
            data = DataBatch(train_data_dir, test_data_dir, image_size, label_size)
            data.prepare_data(params.batch_size, capacity_of_train, params.test_size, capacity_of_test)  # 准备训练和测试数据
            train_model = TrainModel(ckpt_dir, graph_dir, log_dir)
            # 训练模型
            train_model.model_training(params, data)
            
            break
        else:
            print("Wrong input!")

