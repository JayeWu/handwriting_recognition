import tensorflow as tf
import numpy as np
import os
import datetime
from PIL import Image
import matplotlib.pyplot as plt
from tensorflow.python.platform import gfile


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
class InitData:
    def __init__(self):
        self.dir_train = "data/train.tfrecords"
        self.dir_test = "data/test.tfrecords"
        self.image_batch = None
        self.label_batch = None
        self.image_batch_test = None
        self.label_batch_test = None

    # 准备训练和测试数据
    def prepare_data(self, batch_size, capacity, test_data_size, test_capacity):
        reader = tf.TFRecordReader()  # reader for TFRecord file
        test_queue = tf.train.string_input_producer([self.dir_test])  # files read queue
        train_queue = tf.train.string_input_producer([self.dir_train])
        _, serialized_example = reader.read(train_queue)  # examples in TFRecord file
        _, serialized_test = reader.read(test_queue)
        features = parse_image_example(serialized_example)  # parsing features
        features_test = parse_image_example(serialized_test)
        image = tf.decode_raw(features['image_raw'], tf.uint8)  # decode image data from string to image, a Tensor
        image.set_shape([784])  # pixels is 784
        label = tf.cast(features['label'], tf.int32)
        image_test = tf.decode_raw(features_test['image_raw'], tf.uint8)
        image_test.set_shape([784])
        label_test = tf.cast(features_test['label'], tf.int32)
        self.image_batch, lb = tf.train.shuffle_batch(
            [image, label], batch_size=batch_size, capacity=capacity, min_after_dequeue=500)  # queue of image_batch, shuffle_batch mean random
        self.label_batch = tf.one_hot(lb, 10)  # one_hot, 2 for [0,0,1,0,0,0,...]
        self.image_batch_test, lb_test = tf.train.shuffle_batch(
            [image_test, label_test], batch_size=test_data_size, capacity=test_capacity, min_after_dequeue=0)
        self.label_batch_test = tf.one_hot(lb_test, 10)


# 超参数 类
class Params:
    def __init__(self):
        self.batch_size = 128  # batch size of train data
        self.test_size = 256  # test data size 一次测试所用的数据量
        self.learning_rate = 0.001
        self.max_epochs = 10000
        self.expect_accuracy = 0.98

    # 输入超参数
    def input_params(self):
        self.batch_size = int(input("please enter the  batch size of train data (128): "))
        self.test_size = int(input("please enter the size of test data (1000): "))
        self.learning_rate = float(input("please enter the learning rate of RMS optimizer (0.001): "))
        self.max_epochs = int(input("please enter the maximal epochs of model training (10000): "))
        self.expect_accuracy = float(input("please enter the expect accuracy of model (0.99): "))


# 训练模型
class TrainModel:
    def __init__(self, ckpt_dir):
        self.ckpt_dir = ckpt_dir

    # 训练的方法
    def model_training(self, image_batch, label_batch, image_batch_test, label_batch_test, max_epochs,
                       expect_accuracy, learning_rate):

        X = tf.placeholder("float", [None, 28, 28, 1], name="X")  # placeholder for model input(image data)
        Y = tf.placeholder("float", [None, 10], name="Y")  # placeholder for model output (prediction)
        
        w = init_weights([3, 3, 1, 6])  # weights of first level
        w2 = init_weights([3, 3, 6, 16])
        w3 = init_weights([3, 3, 16, 32])
        w4 = init_weights([32*4*4, 128])
        w_o = init_weights([128, 10])

        if not os.path.exists(self.ckpt_dir):
            os.makedirs(self.ckpt_dir)
        saver = tf.train.Saver()  # saver for model

        p_keep_conv = tf.placeholder("float", name="p_keep_conv")
        p_keep_hidden = tf.placeholder("float", name="p_keep_hidden")
        py_x = model(X, w, w2, w3, w4, w_o, p_keep_conv, p_keep_hidden)  # 构建网络模型, 输出为py_x

        cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=py_x, labels=Y))  # 代价函数
        train_op = tf.train.RMSPropOptimizer(learning_rate, 0.9).minimize(cost)  # 训练操作 使用RMS优化器
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
            epochs = max_epochs

            for i in range(max_epochs):
                '''训练
                每次训练取出一个批次的数据，喂给train_op
                为了动态展示训练进程， 每次训练都打印出模型预测准确度
                '''
                bx, batch_ys = sess.run([image_batch, label_batch])
                batch_xs = bx.reshape(-1, 28, 28, 1)
                sess.run(train_op, feed_dict={X: batch_xs, Y: batch_ys, p_keep_conv: 0.8, p_keep_hidden: 0.5})
                tex, tey = sess.run([image_batch_test, label_batch_test])
                tex = tex.reshape(-1, 28, 28, 1)
                accuracy = np.mean(np.argmax(tey, axis=1) ==
                                   sess.run(predict_op, feed_dict={X: tex, p_keep_conv: 1.0, p_keep_hidden: 1.0}))
                print("The accuracy after %d train epochs is %.4f" % (i, accuracy))
                if accuracy >= expect_accuracy:
                    epochs = i+1
                    break
            tf.train.write_graph(sess.graph_def, "./graph", "train.pb", False)
            saver.save(sess, self.ckpt_dir + "/model.ckpt")
            coord.request_stop()
            coord.join(threads)
            end_time = datetime.datetime.now()
            duration = end_time - start_time
            print("accuracy: %2.2f%% training duration: %2d seconds  epochs: %2d "
                  % (accuracy*100, duration.seconds, epochs))


# 初始化化权重
def init_weights(shape):
    return tf.Variable(tf.random_normal(shape, stddev=0.01))


# 神经网络结构
def model(X, w, w2, w3, w4, w_o, p_keep_conv, p_keep_hidden):
    l1a = tf.nn.relu(tf.nn.conv2d(X, w, strides=[1, 1, 1, 1], padding='SAME'))
    print(l1a)
    l1 = tf.nn.max_pool(l1a, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    l1 = tf.nn.dropout(l1, p_keep_conv)
    print(l1)

    l2a = tf.nn.relu(tf.nn.conv2d(l1, w2, strides=[1, 1, 1, 1], padding='SAME'))
    print(l2a)
    l2 = tf.nn.max_pool(l2a, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    l2 = tf.nn.dropout(l2, p_keep_conv)
    print(l2)

    l3a = tf.nn.relu(tf.nn.conv2d(l2, w3, strides=[1, 1, 1, 1], padding='SAME'))
    print(l3a)
    l3 = tf.nn.max_pool(l3a, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    l3 = tf.reshape(l3, [-1, w4.get_shape().as_list()[0]])
    l3 = tf.nn.dropout(l3, p_keep_conv)
    print(l3)

    l4 = tf.nn.relu(tf.matmul(l3, w4))
    l4 = tf.nn.dropout(l4, p_keep_hidden)
    print(l4)
    pyx = tf.matmul(l4, w_o)
    return pyx


# 识别图片
def recognize_image(ckpt_dir, dir_be):
    # 手写图片预处理
    dir_af = "pic_for_predict"
    files1 = os.listdir(dir_af)
    for f in files1:
        os.remove(dir_af+"/"+f)
    files = os.listdir(dir_be)
    cnt = len(files)
    for i in range(cnt):
        img = Image.open(dir_be + "/" + files[i])
        img = img.resize((28, 28))
        img = img.convert("L")
        img.save(dir_af + "/" + files[i], "PNG")

    with tf.Session() as sess:
        ''' 加载保存好的图
        如果不进行这一步，也可以按照训练模型时一样的定义需要的张量
        '''
        with gfile.FastGFile("./graph/train.pb", "rb") as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())
            sess.graph.as_default()
            tf.import_graph_def(graph_def, name='')
        predict_op = sess.graph.get_tensor_by_name("predict_op:0")  # 取出预测值的张量
        X = sess.graph.get_tensor_by_name("X:0")  # 取出输入数据的张量
        p_keep_conv = sess.graph.get_tensor_by_name("p_keep_conv:0")  #
        p_keep_hidden = sess.graph.get_tensor_by_name("p_keep_hidden:0")  #

        # 加载保存好的模型
        saver = tf.train.import_meta_graph(ckpt_dir+"/model.ckpt.meta")

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
            files[i] = dir_af+"/"+files[i]
            img = Image.open(files[i])  # 读取要识别的图片
            print("input: ", files[i])
            imga = np.array(img).reshape(-1, 28, 28, 1)
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
        print("Verification accuracy is %.2f" % (correct/cnt))


def random_test_model(ckpt_dir):
    with tf.Session() as sess:
        ''' 加载保存好的图
        如果不进行这一步，也可以按照训练模型时一样的定义需要的张量
        '''
        with gfile.FastGFile("./graph/train.pb", "rb") as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())
            sess.graph.as_default()
            tf.import_graph_def(graph_def, name='')
        predict_op = sess.graph.get_tensor_by_name("predict_op:0")  # 取出预测值的张量
        X = sess.graph.get_tensor_by_name("X:0")  # 取出输入数据的张量
        p_keep_conv = sess.graph.get_tensor_by_name("p_keep_conv:0")  #
        p_keep_hidden = sess.graph.get_tensor_by_name("p_keep_hidden:0")  #
        image_batch_test = sess.graph.get_tensor_by_name("shuffle_batch_1:0")
        label_batch_test = sess.graph.get_tensor_by_name("one_hot_1:0")
        # 加载保存好的模型
        saver = tf.train.import_meta_graph(ckpt_dir+"/model.ckpt.meta")

        ckpt = tf.train.get_checkpoint_state(ckpt_dir)
        if ckpt and ckpt.model_checkpoint_path:
            print(ckpt.model_checkpoint_path)
            saver.restore(sess, ckpt.model_checkpoint_path)
        # 开始识别图片
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)  # 启动队列
        tex, tey = sess.run([image_batch_test, label_batch_test])
        tex = tex.reshape(-1, 28, 28, 1)
        accuracy = np.mean(np.argmax(tey, axis=1) ==
                           sess.run(predict_op, feed_dict={X: tex, p_keep_conv: 1.0, p_keep_hidden: 1.0}))
        coord.request_stop()
        coord.join(threads)
        print("The accuracy is %.4f" % accuracy)


if __name__ == "__main__":
    ckpt_dir = "./ckpt_dir3"  # check point
    verify_pic_dir = "./test_num"  # 识别验证的文件路径
    while 1:
        pattern = input("Please choose training pattern or recognizing or random test pattern (t/r/rt): ")
        if pattern == "r" or pattern == "R":
            recognize_image(ckpt_dir, verify_pic_dir)
            break
        elif pattern == "t" or pattern == "T":
            params = Params()  # 初始化超参数
            params.input_params()  # 输入超参数
            data = InitData()
            data.prepare_data(params.batch_size, 1000, params.test_size, 10000)  # 准备训练和测试数据
            train_model = TrainModel(ckpt_dir)
            # 训练模型
            train_model.model_training(data.image_batch, data.label_batch, data.image_batch_test, data.label_batch_test,
                                       params.max_epochs, params.expect_accuracy, params.learning_rate)

            break
        elif pattern == "rt":
            random_test_model(ckpt_dir)
        else:
            print("Wrong input!")

