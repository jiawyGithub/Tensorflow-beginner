# coding=utf-8
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import faulthandler
# 在import之后直接添加以下启用代码即可
faulthandler.enable()
# 后边正常写你的代码
import traceback

#定义超参
lr = 0.1
num_epoch = 2000
 
#使用numpy生成200个随机点
x_data=np.linspace(-0.5,0.5,100)[:,np.newaxis] #生成的数据放到第一维（:处），np.newaxis的作用是增加一个维度
# [[-0.5       ],[-0.38888889],[-0.27777778]...]
noise=np.random.normal(0,0.02,x_data.shape) # (200, 1) 从正态分布中采样200个点
y_data=np.square(x_data)+noise # np.square就是平方

#定义两个placeholder
x=tf.placeholder(tf.float32,[None,1]) # 任意行1列
y=tf.placeholder(tf.float32,[None,1])
 
#定义神经网络中间层
Weights_L1=tf.Variable(tf.random_normal([1,10])) # shape (1, 10) 权重连接输入层（1维）和中间层（10维），
biases_L1=tf.Variable(tf.zeros([1,10])) # 加到中间层上
Wx_plus_b_L1=tf.matmul(x,Weights_L1)+biases_L1
L1=tf.nn.tanh(Wx_plus_b_L1) # 激活函数
 
#定义神经网络输出层
Weights_L2=tf.Variable(tf.random_normal([10,1])) # 连接中间层（10维）和输出层（1维）
biases_L2=tf.Variable(tf.zeros([1,1])) # 加到输出层上
Wx_plus_b_L2=tf.matmul(L1,Weights_L2)+biases_L2
prediction=tf.nn.tanh(Wx_plus_b_L2) # 激活函数
 
#二次代价函数
loss=tf.reduce_mean(tf.square(y-prediction))
#使用梯度下降法训练
train_step=tf.train.GradientDescentOptimizer(lr).minimize(loss)
 
with tf.Session() as sess:
    #变量初始化
    sess.run(tf.global_variables_initializer())
    for step in range(num_epoch):
        sess.run(train_step,feed_dict={x:x_data,y:y_data})
    #获取预测值 测试时只传x_data就可以了
    prediction_value=sess.run(prediction,feed_dict={x:x_data})

#画图
plt.figure(figsize=(10, 5))
plt.plot(x_data,y_data,'ro')
plt.plot(x_data,prediction_value,'r-')

# plt.scatter(x_data,y_data)
# plt.plot(x_data,prediction_value,'r-',lw=5)
# plt.show()

def test_segmentation_fault():
    # 对于segmentation fault并不能catch到异常，即此处try没效果
    try:
        plt.show()
    except Exception as e:
        print(traceback.format_exc())


if __name__ == "__main__":
    test_segmentation_fault()

