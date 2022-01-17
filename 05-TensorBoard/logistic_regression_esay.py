from csv import writer
from heapq import merge
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
 
#载入数据集
mnist=input_data.read_data_sets("../datasets/MNIST", one_hot=True) # 路径

# 设置超参
batch_size=100 # 每个批次的大小
n_batch=mnist.train.num_examples // batch_size # 计算一共有多少批次； "/"表示浮点数除法，返回浮点结果; "//"表示整数除法。
lr = 0.2 
epoch_num = 20

# 参数概要
def variable_summaries(var):
    with tf.name_scope("summaries"):
        mean = tf.reduce_mean(var)
        tf.summary.scalar('mean', mean) # 平均值
        with tf.name_scope('stddev'):
            stddev = tf.sqrt(tf.reduce_mean(tf.square(var-mean)))
        tf.summary.scalar('stddev', stddev) # 标准差
        tf.summary.scalar('max', tf.reduce_max(var)) # 最大值
        tf.summary.scalar('min', tf.reduce_min(var)) # 最小值
        tf.summary.histogram('histogram', var) # 直方图

# 命名空间
with tf.name_scope("input"):
    #定义两个placeholder
    x=tf.placeholder(tf.float32, [None,784], name="x-input") # 784=28*28
    y=tf.placeholder(tf.float32,[None,10], name="y-input")
 
#创建一个简单的神经网络
with tf.name_scope("layer"):
    with tf.name_scope("wights"):
        W=tf.Variable(tf.zeros([784,10]),name="W")
        variable_summaries(W)
    with tf.name_scope("biases"): 
        b=tf.Variable(tf.zeros([1,10]),name="b")
        variable_summaries(b)
    with tf.name_scope("wx_plus_b"):
        wx_plus_b = tf.matmul(x,W)+b
    with tf.name_scope("softmax"):
        prediction=tf.nn.softmax(wx_plus_b)
 
#二次代价函数
# loss=tf.reduce_mean(tf.square(y-prediction))
# 交叉熵函数
with tf.name_scope("loss"):
    loss=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=y, logits=prediction))
    tf.summary.scalar('loss', loss)
#使用剃度下降法
with tf.name_scope("train"):
    train_step=tf.train.GradientDescentOptimizer(lr).minimize(loss)
 
#初始化变量
init=tf.global_variables_initializer()
 
#结果存放在一个布尔型列表中
with tf.name_scope("accuracy"):

    with tf.name_scope("correct_prediction"):
        correct_prediction=tf.equal(tf.argmax(y,1), tf.argmax(prediction,1)) #argmax返回一维张量中最大的值所在的位置
        # tf.equal，判断，x, y 是不是相等,逐个元素进行判断，如果相等就是True，不相等，就是False。
    
    with tf.name_scope("accuracy"):
        #求准确率
        accuracy=tf.reduce_mean(tf.cast(correct_prediction,tf.float32))
        # tf.cast()函数的作用是执行 tensorflow 中张量数据类型转换，这里把ture/flase转换成1./0.
        tf.summary.scalar('accuracy', accuracy)

# 合并所有的summary
merge = tf.summary.merge_all()

# 下面这些东西跟结构没有关系
with tf.Session() as sess:
    sess.run(init)
    wirter = tf.summary.FileWriter('logs/', sess.graph) # 打印日志
    for epoch in range(epoch_num):
        for batch in range(n_batch):
            batch_xs,batch_ys=mnist.train.next_batch(batch_size)
            summary, _ = sess.run([merge,train_step],feed_dict={x:batch_xs, y:batch_ys})
        
        wirter.add_summary(summary,epoch)
        acc=sess.run(accuracy,feed_dict={x:mnist.test.images, y:mnist.test.labels})
        print("Iter"+str(epoch)+",Testing Accuracy "+str(acc))