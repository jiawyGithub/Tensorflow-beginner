# coding=utf-8
import tensorflow as tf

m1=tf.constant([[3,3]])   #创建一个常量op
m2=tf.constant([[2],[3]])
product=tf.matmul(m1,m2)    #创建一个矩阵乘法op，把m1和m2传入
print(product)    #这里将输出一个Tensor ??? 这个计算需要在session中执行

#定义一个会话1，启动默认图
sess=tf.Session()  #旧版本
result=sess.run(product)    #调用run方法来执行矩阵乘法op,触发了图中的3个op
print(result)
sess.close()

#另一种定义会话的方式（常用），不需要关闭操作
with tf.Session() as sess:
    result=sess.run(product)
    print(result)