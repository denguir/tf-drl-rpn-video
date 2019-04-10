import tensorflow as tf 
import numpy as np 
from tensorflow.python import pywrap_tensorflow

weights3d_dir = '/home/vador/Documents/project/AI/drl-rpn-tf/output-weights/drl-rpn-paris-video/output/vgg16_drl_rpn/paris_train/'
#weights3d_dir = '/home/vador/Documents/project/AI/drl-rpn-tf/data/pre-trained/data3D/drl-rpn-voc2007-2012-trainval/'
weights2d_dir = '/home/vador/Documents/project/AI/drl-rpn-tf/data/pre-trained/drl-rpn-voc2007-2012-trainval/'

model2d = weights2d_dir + 'vgg16_drl_rpn_iter_110000.ckpt'
meta2d = model2d + '.meta'
model3d = weights3d_dir + 'vgg16_drl_rpn_iter_400.ckpt'
meta3d = model3d + '.meta'

v1 = tf.get_variable('hh_weights_video', shape=[2,3,3,300,300])

with tf.Session() as sess:
    #saver = tf.train.import_meta_graph(meta3d)
    #saver.restore(sess, model3d)
    #graph = tf.get_default_graph()
    #sess.run(tf.global_variables_initializer())

    #var = graph.get_tensor_by_name('xr_weights_base_video:0')
    tf.train.Saver().restore(sess, model3d)
    print("v1 : %s" % v1.shape)
    print("v1 : %s" % v1.eval())
    
    
