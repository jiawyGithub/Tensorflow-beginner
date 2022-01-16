import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
 
#载入数据集
mnist=input_data.read_data_sets("../../datasets/MNIST", one_hot=True) # 路径
 
# 设置超参
batch_size=100 # 每个批次的大小
n_batch=mnist.train.num_examples // batch_size # 计算一共有多少批次； "/"表示浮点数除法，返回浮点结果; "//"表示整数除法。
lr = 0.2 
epoch_num = 5

#定义两个placeholder
x=tf.placeholder(tf.float32,[None,784]) # 784=28*28
y=tf.placeholder(tf.float32,[None,10])
 
#创建一个简单的神经网络
W=tf.Variable(tf.zeros([784,10]))
b=tf.Variable(tf.zeros([1,10]))
prediction=tf.nn.softmax(tf.matmul(x,W)+b)
 
#二次代价函数
# loss=tf.reduce_mean(tf.square(y-prediction))
# 交叉熵函数
loss=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=y, logits=prediction))
#使用剃度下降法
train_step=tf.train.GradientDescentOptimizer(lr).minimize(loss)
 
#初始化变量
init=tf.global_variables_initializer()
 
#结果存放在一个布尔型列表中
correct_prediction=tf.equal(tf.argmax(y,1), tf.argmax(prediction,1)) #argmax返回一维张量中最大的值所在的位置
# tf.equal，判断，x, y 是不是相等,逐个元素进行判断，如果相等就是True，不相等，就是False。

#求准确率
accuracy=tf.reduce_mean(tf.cast(correct_prediction,tf.float32))
# tf.cast()函数的作用是执行 tensorflow 中张量数据类型转换，这里把ture/flase转换成1./0.
 
with tf.Session() as sess:
    sess.run(init)
    for epoch in range(epoch_num):
        for batch in range(n_batch):
            batch_xs,batch_ys=mnist.train.next_batch(batch_size)
            sess.run(train_step,feed_dict={x:batch_xs, y:batch_ys})
 
        acc=sess.run(accuracy,feed_dict={x:mnist.test.images, y:mnist.test.labels})
        print("Iter"+str(epoch)+",Testing Accuracy "+str(acc))