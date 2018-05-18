import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets('/tmp/data/',one_hot=True)

n_classes = 10
batch_size = 128
tot_epochs = 10

x = tf.placeholder('float',[None,784])
y = tf.placeholder('float')
drop_rate = 0.4

def convolutional_neural_network(x):
    x = tf.reshape(x,[-1,28,28,1])
    
    con_l1 = tf.layers.conv2d(inputs=x,filters=32,kernel_size=5,padding='same',activation=tf.nn.relu)
    pool_l1 = tf.layers.max_pooling2d(inputs=con_l1,pool_size=2,strides=2,padding='same')
    con_l2 = tf.layers.conv2d(inputs=pool_l1,filters=64,kernel_size=5,padding='same',activation=tf.nn.relu)
    pool_l2 = tf.layers.max_pooling2d(inputs=con_l2,pool_size=2,strides=2,padding='same')
    pool_l2 = tf.reshape(pool_l2,[-1,7*7*64])
    dense = tf.layers.dense(inputs=pool_l2,units=1024,activation=tf.nn.relu)
    drop_out = tf.layers.dropout(inputs=dense,rate=drop_rate,training=True)
    output = tf.layers.dense(inputs=drop_out,units=10,activation=tf.nn.relu)
    return output

def neural_network_training(x):
    prediction = convolutional_neural_network(x)
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction,labels=y))
    optimizer = tf.train.AdamOptimizer().minimize(loss)
    
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for epoch in range(tot_epochs):
            epoch_loss = 0
            for i in range(int(mnist.train.num_examples/batch_size)):
                epoch_x,epoch_y = mnist.train.next_batch(batch_size)
                j,c = sess.run([optimizer,loss],feed_dict={x:epoch_x,y:epoch_y})
                epoch_loss += c
            print('Epoch ',epoch,' completed.Epoch loss = ',epoch_loss)
        correct = tf.equal(tf.argmax(prediction,1),tf.argmax(y,1))
        accuracy = tf.reduce_mean(tf.cast(correct,'float'))
        print('Accuracy: ',accuracy.eval({x:mnist.test.images,y:mnist.test.labels}))
        
neural_network_training(x)        
    
