import tensorflow as tf 
import numpy as np 
from tensorflow.python import pywrap_tensorflow

weights_dir = '/home/vador/Documents/project/AI/drl-rpn-tf/data/pre-trained/drl-rpn-voc2007-2012-trainval-plus-2007test/'
weights3d_dir = '/home/vador/Documents/project/AI/drl-rpn-tf/output-weights/drl-rpn-paris-video/output/vgg16_drl_rpn/paris_train/'

model = weights_dir + 'vgg16_2012_drl_rpn_iter_110000.ckpt'
meta = model + '.meta'

model3d = weights3d_dir + 'vgg16_2012_drl_rpn_iter_0.ckpt'
meta3d = model3d + '.meta'

weight_names = ['xr_weights_base', 'xr_weights_aux', 
                'xh_weights_base', 'xh_weights_aux',
                'xz_weights_base', 'xz_weights_aux',
                'hr_weights', 'hh_weights', 'hz_weights', 'h_relu_weights',
                'additional_weights']

              
with tf.Session() as sess:
    dims_time = 2 # see config file
    # restore weights in golbal_variables
    saver = tf.train.import_meta_graph(meta)
    saver.restore(sess, model)
    graph = tf.get_default_graph()
    # prepare saver for 3d weights
    saver3d = tf.train.Saver()
    # initialize all variables
    sess.run(tf.global_variables_initializer())

    weigths_vid = []
    for w_name in weight_names:
        w_tf = graph.get_tensor_by_name(w_name + ":0")
        w_np = sess.run(w_tf)
        w_vid_np = np.stack(dims_time * [w_np], axis=0)
        w_vid_tf = tf.Variable(w_vid_np, name=w_name+'_video')
        print(w_vid_tf.name, '\t', w_vid_tf.shape)
        # initialize new variable
        sess.run(w_vid_tf.initializer) 
        weigths_vid.append(w_vid_tf)
    sess.run(weigths_vid)
    saver3d.save(sess, model3d)
