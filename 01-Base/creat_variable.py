# coding=utf-8
import tensorflow as tf
 
x=tf.Variable([1,2])
a=tf.constant([3,3])
 
sub=tf.subtract(x,a)    # 增加一个减法op
add=tf.add(x,sub)   # 增加一个加法op
 
# 注意：变量在使用之前要在sess中做初始化，但是下边这种初始化方法不会指定变量的初始化顺序
init=tf.global_variables_initializer() # 全局变量初始化
with tf.Session() as sess:
    sess.run(init) 
    print(sess.run(sub))
    print(sess.run(add))
 
 
#################分割线#####################

# 创建一个名字为‘counter’的变量 初始化0
state=tf.Variable(0,name='counter')
new_value=tf.add(state,1)   # 创建一个op，作用是使state加1
update=tf.assign(state,new_value)  # 赋值op；tensorflow不是用python实现的，是c++实现的，因此如果想把python中的值在session中修改，必须要用这种接口的方式
init=tf.global_variables_initializer()
 
with tf.Session() as sess:
    sess.run(init)
    print(sess.run(state))
    for _ in range(5):
        sess.run(update)
        print(sess.run(state))