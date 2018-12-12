import tensorflow as tf


class BinaryModel:
    def __init__(self, name='noname'):
        self.x = tf.placeholder(tf.float32, [None, 128, 128, 3])
        self.y = tf.placeholder(tf.float32, [None])

        self.name = name
        self.build_network()
        var_list = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.name)
        self.saver = tf.train.Saver(var_list)

    def build_network(self):
        h = self.x
        with tf.variable_scope(self.name):
            with tf.variable_scope('layer1'):
                h = tf.contrib.layers.conv2d(h, 16, (5, 5))
                h = tf.nn.relu(h)
                h = tf.contrib.layers.conv2d(h, 16, (5, 5))
                h = tf.nn.relu(h)
                h = tf.contrib.layers.max_pool2d(h, (2, 2))

            with tf.variable_scope('layer2'):
                h = tf.contrib.layers.conv2d(h, 32, (5, 5))
                h = tf.nn.relu(h)
                h = tf.contrib.layers.conv2d(h, 32, (5, 5))
                h = tf.nn.relu(h)
                h = tf.contrib.layers.max_pool2d(h, (2, 2))

            with tf.variable_scope('layer3'):
                h = tf.contrib.layers.conv2d(h, 64, (5, 5))
                h = tf.nn.relu(h)
                h = tf.contrib.layers.conv2d(h, 64, (5, 5))
                h = tf.nn.relu(h)
                h = tf.contrib.layers.max_pool2d(h, (2, 2))

            with tf.variable_scope('layer4'):
                h = tf.contrib.layers.conv2d(h, 128, (5, 5))
                h = tf.nn.relu(h)
                h = tf.contrib.layers.conv2d(h, 128, (5, 5))
                h = tf.nn.relu(h)
                h = tf.contrib.layers.max_pool2d(h, (2, 2))

            with tf.variable_scope('layer5'):
                h = tf.contrib.layers.flatten(h)
                h = tf.contrib.layers.fully_connected(h, 128)

            with tf.variable_scope('layer6'):
                logit = tf.contrib.layers.fully_connected(h, 1, activation_fn=None)
                probs = tf.nn.sigmoid(logit)

                self.probs = probs
                self.logit = logit

        self.loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.expand_dims(self.y, -1), logits=logit))
        self.accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.greater(tf.squeeze(self.probs), 0.5),
                                                        tf.greater(self.y, 0.5)), tf.float32))
        self.train_op = tf.train.AdamOptimizer().minimize(self.loss)

    def main_train(self, dataset):
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        with tf.Session(config=config) as sess:
            sess.run(tf.global_variables_initializer())
            for i, (x_batch, y_batch) in enumerate(dataset):
                _, acc, loss = sess.run([self.train_op, self.accuracy, self.loss],
                                        feed_dict={self.x: x_batch, self.y: y_batch})
                print('\rBatch %d, acc: %.3f, loss: %.3f' % (i, acc, loss), end='')

                if i % 1000 == 0:
                    self.save(sess)
                    print()

    def save(self, sess):
        self.saver.save(sess, './ckpts/%s' % self.name)

    def load(self, sess):
        self.saver.restore(sess, './ckpts/%s' % self.name)


class ClassifyModel:
    def __init__(self, name='noname'):
        self.x = tf.placeholder(tf.float32, [None, 128, 128, 3])
        self.y = tf.placeholder(tf.float32, [None, 4])

        self.name = name
        self.build_network()
        var_list = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.name)
        self.saver = tf.train.Saver(var_list)

    def build_network(self):
        h = self.x
        f = tf.nn.leaky_relu
        with tf.variable_scope(self.name):
            with tf.variable_scope('layer1'):
                h = tf.contrib.layers.conv2d(h, 64, (5, 5), activation_fn=f)
                h = tf.contrib.layers.conv2d(h, 64, (5, 5), activation_fn=f)
                h = tf.contrib.layers.max_pool2d(h, (2, 2))

            with tf.variable_scope('layer2'):
                h = tf.contrib.layers.conv2d(h, 64, (5, 5), activation_fn=f)
                h = tf.contrib.layers.conv2d(h, 64, (5, 5), activation_fn=f)
                h = tf.contrib.layers.max_pool2d(h, (2, 2))

            with tf.variable_scope('layer3'):
                h = tf.contrib.layers.conv2d(h, 64, (5, 5), activation_fn=f)
                h = tf.contrib.layers.conv2d(h, 64, (5, 5), activation_fn=f)
                h = tf.contrib.layers.max_pool2d(h, (2, 2))

            with tf.variable_scope('layer4'):
                h = tf.contrib.layers.conv2d(h, 128, (5, 5), activation_fn=f)
                h = tf.contrib.layers.conv2d(h, 128, (5, 5), activation_fn=f)
                h = tf.contrib.layers.max_pool2d(h, (2, 2))

            with tf.variable_scope('layer5'):
                h = tf.contrib.layers.flatten(h)
                h = tf.contrib.layers.fully_connected(h, 128, activation_fn=f)

            with tf.variable_scope('layer6'):
                logit = tf.contrib.layers.fully_connected(h, 4, activation_fn=None)
                probs = tf.nn.softmax(logit)

                self.probs = probs
                self.logit = logit

        self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=self.y, logits=logit))
        self.accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(logit),
                                                        tf.argmax(self.y)),
                                               tf.float32)
                                       )
        self.train_op = tf.train.AdamOptimizer().minimize(self.loss)

    def main_train(self, dataset):
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        with tf.Session(config=config) as sess:
            sess.run(tf.global_variables_initializer())
            for i, (x_batch, y_batch) in enumerate(dataset):
                _, acc, loss = sess.run([self.train_op, self.accuracy, self.loss],
                                        feed_dict={self.x: x_batch, self.y: y_batch})
                print('\rBatch %d, acc: %.3f, loss: %.3f' % (i, acc, loss), end='')

                if i % 1000 == 0:
                    self.save(sess)
                    print()

    def save(self, sess):
        self.saver.save(sess, './ckpts/%s' % self.name)

    def load(self, sess):
        self.saver.restore(sess, './ckpts/%s' % self.name)
