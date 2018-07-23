import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import tensorflow as tf

a = tf.constant(2)
b = tf.constant(4)
x = tf.add(a, b)
writer = tf.summary.FileWriter('../graphs', tf.get_default_graph())
with tf.Session() as sess:
    print(sess.run(x))
# close the writer when you’re done using it
writer.close()
