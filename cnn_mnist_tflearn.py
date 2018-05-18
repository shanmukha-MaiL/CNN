import tflearn
from tflearn.layers.conv import conv_2d,max_pool_2d
from tflearn.layers.core import input_data,dropout,fully_connected
from tflearn.layers.estimator import regression
import tflearn.datasets.mnist as mnist
import numpy as np

x,y,test_x,test_y = mnist.load_data(one_hot=True)
x = tflearn.reshape(x,[-1,28,28,1])
test_x = tflearn.reshape(test_x,[-1,28,28,1])

conv_net = input_data([None,28,28,1],name='input')

conv_net = conv_2d(conv_net,32,5,activation='relu')
conv_net = max_pool_2d(conv_net,2,2)
conv_net = conv_2d(conv_net,64,5,activation='relu')
conv_net = max_pool_2d(conv_net,2,2)
conv_net = fully_connected(conv_net,1024,activation='relu')
conv_net = dropout(conv_net,0.8)
conv_net = fully_connected(conv_net,10,activation='softmax')
conv_net = regression(conv_net,optimizer='adam',learning_rate=0.01,loss='categorical_crossentropy',name='targets')
model = tflearn.DNN(conv_net)
model.fit({'input':x},{'targets':y},n_epoch=10,validation_set=({'input':test_x},{'targets':test_y}),snapshot_step=500,show_metric=True,run_id='mnist')



print(np.round(model.predict(test_x[1])))
print(test_y[1])


