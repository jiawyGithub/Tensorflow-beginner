# coding=utf-8
import tensorflow as tf
import numpy as np

x_data=np.random.rand(100) #使用numpy生成100个随机点s
y_data=x_data*0.1+0.2   #这里我们设定已知直线的k为0.1 b为0.2得到y_data

# 定义超参
lr = 0.2
epoch_num = 200

#构造一个线性模型
b=tf.Variable(0.)
k=tf.Variable(0.)
y=k*x_data+b

#二次代价函数（白话：两数之差平方后取 平均值）
loss=tf.reduce_mean(tf.square(y_data-y))
#定义一个梯度下降法来进行训练的优化器（其实就是按梯度下降的方法改变线性模型k和b的值，注意这里的k和b一开始初始化都为0.0，后来慢慢向0.1、0.2靠近）
optimizer=tf.train.GradientDescentOptimizer(lr)    #这里的lr是梯度下降的系数
#最小化代价函数(训练的方式就是使loss值最小，loss值可能是随机初始化100个点与模型拟合出的100个点差的平方相加...等方法)
train=optimizer.minimize(loss)

#初始化变量
init=tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    for step in range(epoch_num):
        sess.run(train)
        if (step+1)%20==0:
            print('step',step+1,'loss',loss.eval(),'[k,b]',sess.run([k,b])) #这里使用fetch的方式只是打印k、b的值，每20次打印一下，改变k、b的值是梯度下降优化器的工作