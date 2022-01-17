import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
 
#载入数据集
mnist=input_data.read_data_sets("../datasets/MNIST", one_hot=True)

# 设置超参
batch_size=100 # 每个批次的大小
n_batch=mnist.train.num_examples // batch_size # 计算一共有多少批次； "/"表示浮点数除法，返回浮点结果; "//"表示整数除法。
epoch_num = 5
 
#定义两个placeholder
x=tf.placeholder(tf.float32,[None,784])
y=tf.placeholder(tf.float32,[None,10])
keep_prob=tf.placeholder(tf.float32)
lr=tf.Variable(0.001, dtype=tf.float32)
 
#创建一个简单的神经网络
W1=tf.Variable(tf.truncated_normal([784,20],stddev=0.1))   #这里我们使用一个截断的正太分布初始化W
b1=tf.Variable(tf.zeros([1,20]))
L1=tf.nn.tanh(tf.matmul(x,W1)+b1)   #激活函数为双曲正切函数
L1_drop=tf.nn.dropout(L1, keep_prob)
 
W2=tf.Variable(tf.truncated_normal([20,15],stddev=0.1))
b2=tf.Variable(tf.zeros([1,15]))
L2=tf.nn.tanh(tf.matmul(L1_drop, W2)+b2)
L2_drop=tf.nn.dropout(L2, keep_prob)
 
W3=tf.Variable(tf.truncated_normal([15,10], stddev=0.1))
b3=tf.Variable(tf.zeros([1,10]))
prediction=tf.nn.softmax(tf.matmul(L2_drop,W3)+b3)
 
# 二次代价函数
# loss=tf.reduce_mean(tf.square(y-prediction))
# 交叉熵函数
loss=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=y, logits=prediction))
#使用剃度下降法
train_step=tf.train.AdamOptimizer(lr).minimize(loss)
 
#初始化变量
init=tf.global_variables_initializer()
 
#结果存放在一个布尔型列表中
correct_prediction=tf.equal(tf.argmax(y,1), tf.argmax(prediction,1)) #argmax返回一维张量中最大的值所在的位置
#求准确率
accuracy=tf.reduce_mean(tf.cast(correct_prediction,tf.float32))
 
with tf.Session() as sess:
    sess.run(init)
    for epoch in range(epoch_num):
        sess.run(tf.assign(lr, 0.001*(0.95**epoch))) # 令学习率不断下降
        for batch in range(n_batch):
            batch_xs,batch_ys=mnist.train.next_batch(batch_size)
            sess.run(train_step,feed_dict={x:batch_xs, y:batch_ys,keep_prob:1})
 
        test_acc=sess.run(accuracy,feed_dict={x:mnist.test.images, y:mnist.test.labels,keep_prob:1})
        train_acc=sess.run(accuracy,feed_dict={x:mnist.train.images, y:mnist.train.labels,keep_prob:1})
        print("Iter"+str(epoch)+",Testing Accuracy "+str(test_acc)+" Training Accuracy "+str(train_acc))