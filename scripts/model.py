import tensorflow as tf

slim = tf.contrib.slim
layers = tf.contrib.layers

def inference(inputs,
              num_outputs=64,
              is_training=True,
              weight_decay=0.0,
              reuse=None):

    batch_norm_params = {
        # Decay for the moving averages.
        'decay': 0.99,
        # epsilon to prevent 0s in variance.
        'epsilon': 0.001,
	'center': True,
	'scale': True,
    }

    with slim.arg_scope([slim.conv2d],

            weights_initializer=layers.variance_scaling_initializer(factor=1.0,
                                                                    mode='FAN_AVG',
                                                                    uniform=False),
            weights_regularizer=slim.l2_regularizer(weight_decay),
	    activation_fn=tf.nn.relu,
            normalizer_fn=slim.batch_norm,
            normalizer_params=batch_norm_params):

        return model(inputs,
                     num_outputs,
                     is_training=is_training,
                     reuse=reuse)

def model(inputs,
          num_outputs=64,
          is_training=True,
          scope='win37_dep9',
          reuse=None):

    with tf.variable_scope(scope, 'win37_dep9', reuse=reuse):
        with slim.arg_scope([slim.batch_norm],is_training=is_training):
            with slim.arg_scope([slim.conv2d],stride=1, padding='VALID'):

                net = slim.conv2d(inputs, 32, [5, 5], scope='conv1')
                net = slim.conv2d(net, 32, [5, 5], scope='conv2')
                net = slim.conv2d(net, 64, [5, 5], scope='conv3')
                net = slim.conv2d(net, 64, [5, 5], scope='conv4')
                net = slim.conv2d(net, 64, [5, 5], scope='conv5')
                net = slim.conv2d(net, 64, [5, 5], scope='conv6')
                net = slim.conv2d(net, 64, [5, 5], scope='conv7')
                net = slim.conv2d(net, 64, [5, 5], scope='conv8')
                net = slim.conv2d(net, num_outputs, [5, 5], scope='conv9', activation_fn=None, normalizer_fn=None)
                net = slim.batch_norm(net, is_training=is_training)

                return net

