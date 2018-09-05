import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from utils import *

# our target data
X=np.linspace(-10,10,500)
y=np.sin(X)
plt.plot(X,y)
plt.show()
# Split train and test
X_train, X_test=train_test_split(X,test_size=0.8,random_state=10)
Y_train=np.sin(X_train)
Y_test=np.sin(X_test)
plt.scatter(X_test,Y_test)
plt.show()

X_train1=X_train[:,np.newaxis]
Y_train1=Y_train[:,np.newaxis]
X_test1=X_test[:,np.newaxis]
Y_test1=Y_test[:,np.newaxis]
#define train parameters
n_iter=5000
plot_freq=200

X_content=tf.placeholder(tf.float32,shape=[None,1],name='X_content')
Y_content=tf.placeholder(tf.float32,shape=[None,1],name='Y_content')
X_target=tf.placeholder(tf.float32,shape=[None,1],name='X_target')
Y_target=tf.placeholder(tf.float32,shape=[None,1],name='Y_target')


NPR=NP()
train_op, loss=NPR.build_model(X_content,Y_content,X_target,Y_target,learning_rate=0.001)
predict_op=NPR.posterior_pred(X_content,Y_content,X_target)

sess=tf.Session()
init=tf.global_variables_initializer()
sess.run(init)
for iter in range(n_iter):
    X_c, Y_c, X_t, Y_t = train_ct(X_train, Y_train)
    _,l=sess.run([train_op, loss],feed_dict={X_content:X_c,Y_content:Y_c,X_target:X_t,Y_target:Y_t})
    #print(l)

pred_y=sess.run(predict_op,feed_dict={X_content:X_train1,Y_content:Y_train1,X_target:X_test1})
#print(pred_y.shape)
plt.scatter(X_test,pred_y, color='r')
plt.show()






