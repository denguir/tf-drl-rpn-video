  
import tensorflow as tf
import numpy as np

class NetworkVideo():

  def __init__(self):
      self._predictions = {}

  def run(self, sess, rl_in, rl_hid, aux_info):
    done_prob, fix_prob, done_logits, fix_logits, h =  self._conv_gru(is_training=True, initializer=tf.contrib.layers.xavier_initializer())
    init = tf.global_variables_initializer()
    sess.run(init)
    dp = sess.run(done_prob, feed_dict={self._rl_in : rl_in, self._rl_hid : rl_hid, self._aux_done_info : aux_info})
    fp = sess.run(fix_prob, feed_dict={self._rl_in : rl_in, self._rl_hid : rl_hid, self._aux_done_info : aux_info})
    dl = sess.run(done_logits, feed_dict={self._rl_in : rl_in, self._rl_hid : rl_hid, self._aux_done_info : aux_info})
    fl = sess.run(fix_logits, feed_dict={self._rl_in : rl_in, self._rl_hid : rl_hid, self._aux_done_info : aux_info})
    hh = sess.run(h, feed_dict={self._rl_in : rl_in, self._rl_hid : rl_hid, self._aux_done_info : aux_info})

    print('done_prob shape:', dp.shape)
    print('fix_prob shape:', fp.shape)
    print('done_logits shape:', dl.shape)
    print('fix_logits shape:', fl.shape)
    print('hidden state shape:', hh.shape)


  def _make_var(self, name, shape, initializer=None, is_training=True):    
    return tf.get_variable(name, shape, dtype=None, initializer=initializer,
                           regularizer=None, trainable=is_training)
  # The Conv-GRU processor
  def _conv_gru(self, is_training, initializer, name='conv_gru'):
    # Extract some relevant config keys for convenience
    dims_base = 512
    dims_aux = 39
    dims_tot = 551
    dims_time = 2

    # Input placeholders
    # dims: batch-time-height-width-channel-frame_number
    self._rl_in = tf.placeholder(tf.float32, shape=[None, dims_time, None, None, dims_tot])
    self._rl_hid = tf.placeholder(tf.float32, shape=[None, dims_time, None, None, 300])

    # Define convenience operator
    self.conv = lambda i, k: tf.nn.conv2d(i, k, [1, 1, 1, 1], padding='SAME')
    self.conv3d = lambda i, k: tf.nn.conv3d(i, k, [1, 1, 1, 1, 1], padding='SAME')
      
    # Create conv-GRU kernels
    self.xr_kernel_base = self._make_var('xr_weights_base', [dims_time, 3, 3, dims_base, 240],
                                    initializer, is_training)
    self.xr_kernel_aux = self._make_var('xr_weights_aux', [dims_time, 9, 9, dims_aux, 60],
                                   initializer, is_training)
    self.xh_kernel_base = self._make_var('xh_weights_base', [dims_time, 3, 3, dims_base, 240],
                                    initializer, is_training)
    self.xh_kernel_aux = self._make_var('xh_weights_aux', [dims_time, 9, 9, dims_aux, 60],
                                   initializer, is_training)
    self.xz_kernel_base = self._make_var('xz_weights_base', [dims_time, 3, 3, dims_base, 240],
                                    initializer, is_training)
    self.xz_kernel_aux = self._make_var('xz_weights_aux', [dims_time, 9, 9, dims_aux, 60],
                                   initializer, is_training)
    self.hr_kernel = self._make_var('hr_weights', [dims_time, 3, 3, 300, 300],
                               initializer, is_training)
    self.hh_kernel = self._make_var('hh_weights', [dims_time, 3, 3, 300, 300],
                               initializer, is_training)
    self.hz_kernel = self._make_var('hz_weights', [dims_time, 3, 3, 300, 300],
                               initializer, is_training)
    self.h_relu_kernel = self._make_var('h_relu_weights', [dims_time, 3, 3, 300, 128],
                                   initializer, is_training)

    # Create Conv-GRU biases
    bias_init = initializer
    self.r_bias = self._make_var('r_bias', [300], bias_init, is_training)
    self.h_bias = self._make_var('h_bias', [300], bias_init, is_training)
    self.z_bias = self._make_var('z_bias', [300], bias_init, is_training)
    self.relu_bias = self._make_var('relu_bias', [128], bias_init, is_training) 

    # Used for some aux info (e.g. exploration penalty when used as feature)
    add_dim = 130
    self.additional_kernel = self._make_var('additional_weights', [dims_time, 3, 3, add_dim, 2],
                                            initializer, is_training)
    self.additional_bias = self._make_var('additional_bias', [2], bias_init,
                                          is_training)
    self._aux_done_info = tf.placeholder(tf.float32,
                                         shape=[None, dims_time, None, None, 2])

    # Define weights for stopping condition (no bias here)
    self.done_weights = self._make_var('done_weights', [625, 1], initializer,
                                       is_training)

    # We need to make a TensorFlow dynamic graph-style while-loop, as our
    # conv-GRU will be unrolled a different number of steps depending on the
    # termination decisions of the agent
    
    # First we need to set some init / dummy variables
    in_shape = tf.shape(self._rl_in)
    done_logits_all = tf.zeros([0, 1])
    fix_logits_all = tf.zeros([0, in_shape[2] * in_shape[3]])
    done_prob = tf.zeros([0, 1])
    fix_prob_map = tf.zeros([0, 0, 0, 0])
    h = tf.slice(self._rl_hid, [0, 0, 0, 0, 0], [1, -1, -1, -1, -1])
    
    # Looping termination condition (TF syntax demands also the other variables
    # are sent as input, although not used for the condition check)
    nbr_steps = in_shape[0]
    i = tf.constant(0)
    while_cond = lambda i, done_logits_all, fix_logits_all, done_prob, h, fix_prob_map: tf.less(i, nbr_steps)

    # Unroll current step (if forward pass) and if in training unroll
    # a full rollout
    i, done_logits_all, fix_logits_all, done_prob, h, fix_prob_map \
      = tf.while_loop(while_cond, self.rollout, [i, done_logits_all, fix_logits_all, done_prob, h, fix_prob_map],
                             shape_invariants=[i.get_shape(), tf.TensorShape([None, 1]),
                                               tf.TensorShape([None, None]), tf.TensorShape([None, 1]),
                                               h.get_shape(), tf.TensorShape([None, None, None, None])])

    # Insert to containers
    self._predictions['done_prob'] = done_prob
    self._predictions['fix_prob'] = fix_prob_map
    self._predictions['done_logits'] = done_logits_all
    self._predictions['fix_logits'] = fix_logits_all
    self._predictions['rl_hid'] = h

    return done_prob, fix_prob_map, done_logits_all, fix_logits_all, h


  # Unroll the GRU
  def rollout(self, i, done_logits_all, fix_logits_all, done_prob, h, fix_prob_map):

    # Extract some relevant config keys for convenience
    dims_base = 512

    # Split into base feature map and auxiliary input
    rl_base = tf.slice(self._rl_in, [i, 0, 0, 0, 0], [1, -1, -1, -1, dims_base])
    rl_aux = tf.slice(self._rl_in, [i, 0, 0, 0, dims_base], [1, -1, -1, -1, -1])

    # eq. (1)
    xr_conv = tf.concat([self.conv3d(rl_base, self.xr_kernel_base),
                         self.conv3d(rl_aux, self.xr_kernel_aux)], 4)
    hr_conv = self.conv3d(h, self.hr_kernel)
    r = tf.sigmoid(xr_conv + hr_conv + self.r_bias)

    # eq. (2)
    xh_conv = tf.concat([self.conv3d(rl_base, self.xh_kernel_base),
                         self.conv3d(rl_aux, self.xh_kernel_aux)], 4)
    hh_conv = self.conv3d(r * h, self.hh_kernel)
    hbar = tf.tanh(xh_conv + hh_conv + self.h_bias)

    # eq. (3)
    xz_conv = tf.concat([self.conv3d(rl_base, self.xz_kernel_base),
                         self.conv3d(rl_aux, self.xz_kernel_aux)], 4)
    hz_conv = self.conv3d(h, self.hz_kernel)
    z = tf.sigmoid(xz_conv + hz_conv + self.z_bias) 

    # eq. (4)
    h = (1 - z) * h + z * hbar

    # eq. (5)
    conv_gru = tf.nn.relu(self.conv3d(h, self.h_relu_kernel) + self.relu_bias)

    # HERE BEGINS THE CODE FOR TRANSFORMING INTO ACTION PROBABILITIES
    aux_done_info = tf.slice(self._aux_done_info, [i, 0, 0, 0, 0], [1, -1, -1, -1, -1])

    # Extract relevant stuff
    input_shape = tf.shape(conv_gru)
    batch_sz = 1 # must be 1
    time = input_shape[1]
    height = input_shape[2]
    width = input_shape[3]

    # Append beta and time-info (auxiliary info)
    conv_gru \
      = tf.concat([conv_gru,
                   tf.ones((batch_sz, time, height, width, 2)) * aux_done_info], 4)

    # eq. (6)
    conv_gru_processed = tf.nn.tanh(self.conv3d(conv_gru, self.additional_kernel) \
                                    + self.additional_bias)

    done_slice = tf.slice(conv_gru_processed, [0, 0, 0, 0, 0], [1, -1, -1, -1, 1])
    done_slice = tf.reduce_mean(done_slice, 1)
    fix_slice = tf.slice(conv_gru_processed, [0, 0, 0, 0, 1], [1, -1, -1, -1, 1])
    fix_slice = tf.reduce_mean(fix_slice, 1)

    done_slice_reshaped = tf.image.resize_images(done_slice, [25, 25])
    done_slice_vecd = tf.reshape(done_slice_reshaped, [batch_sz, 625])
    done_logits = tf.matmul(done_slice_vecd, self.done_weights) 
    done_prob = tf.sigmoid(done_logits)

    # Probability of where to fix next (need some rearrangement in
    # between to get proper dimensions over which softmax is performed)
    reshape_layer = tf.reshape(tf.transpose(fix_slice, [0, 3, 1, 2]),
                               [1, 1, height * width])
    smax_layer = tf.nn.softmax(reshape_layer)
    fix_prob_map = tf.transpose(tf.reshape(smax_layer,
                              [1, 1, height, width]), [0, 2, 3, 1])
    fix_slice_logits = tf.reshape(fix_slice, [batch_sz, -1]) 

    # Append
    done_logits_all = tf.concat([done_logits_all, done_logits], 0)
    fix_logits_all = tf.concat([fix_logits_all, fix_slice_logits], 0)

    # Return
    return tf.add(i, 1), done_logits_all, fix_logits_all, done_prob, h, fix_prob_map

if __name__ == '__main__':

    dims_base = 512
    dims_tot = 551
    dims_time = 2
    dims_aux = dims_tot - dims_base
    height = 28
    width = 28
    batch = 2

    net = NetworkVideo()
    rl_in = np.random.uniform(
                        low=0,
                        high=10,
                        size=[batch, dims_time, height, width, dims_tot]
                        )
    
    rl_hid = np.random.uniform(
                        low=0,
                        high=10,
                        size=[batch, dims_time, height, width, 300]
                        )

    aux_info = np.random.uniform(
                        low=0,
                        high=10,
                        size=[batch, dims_time, height, width, 2]
                        ) 

    tfconfig = tf.ConfigProto(allow_soft_placement=True)
    tfconfig.gpu_options.allow_growth=True
    # added because cudnn & cublas fail:
    tfconfig.gpu_options.allocator_type = 'BFC'
    tfconfig.gpu_options.per_process_gpu_memory_fraction = 0.9 

    with tf.Session(config=tfconfig) as sess:
      net.run(sess, rl_in, rl_hid, aux_info)
