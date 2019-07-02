import tensorflow as tf


def block(net, n_channel, n_repeat):
    for itr in range(n_repeat):
        net = tf.layers.conv2d(
            net,
            n_channel,
            3,
            activation=tf.nn.relu,
            padding="valid"
        )
    net = tf.layers.max_pooling2d(
        net,
        2, 2
    )
    return net


images = tf.placeholder(tf.float32, [None, 224, 224, 3])

# cnn
net = block(images, 64, 2)
net = block(net, 128, 2)
net = block(net, 256, 3)
net = block(net, 512, 3)
net = block(net, 512, 3)

net = tf.flatten(net)

# fc
net = tf.layers.dense(net, 4096, activation=tf.nn.relu)
net = tf.layers.dense(net, 4096, activation=tf.nn.relu)
net = tf.layers.dense(net, 1000, activation=tf.nn.relu)
net = tf.layers.dense(net, 6, activation=None)
