import tensorflow as tf
import numpy as np

class AutoEncoder:
  def __init__(self, configs, from_zero=True):
    self.input_size = configs['input_size']

    self.filters_number = configs['filters_number']
    self.layers_number = len(self.filters_number)
    self.filters_size = configs['filters_size']

    if from_zero:
      self.build_model()
    else:
      self.load_model(configs['model_path'])

    self.init_session()
    self.warmup()

  def load_model(self, path):
    pass

  def build_model(self):
    self.input = tf.placeholder(tf.float32, (None, self.input_size[0], self.input_size[1] , 1), name='input')
    self.target = tf.placeholder(tf.float32, (None, self.input_size[0], self.input_size[1] , 1), name='target')

    current_input = self.input
    sizes = []

    # Encode the image
    for i in range(self.layers_number):
      conv = tf.layers.conv2d(inputs=current_input,
                              filters=self.filters_number[i],
                              kernel_size=self.filters_size[i],
                              padding='same',
                              activation=tf.nn.relu,
                              name='Conv_{}'.format(i+1))
      sizes.append(conv.shape)
      maxpool = tf.layers.max_pooling2d(conv,
                                        pool_size=(2, 2),
                                        strides=(2, 2),
                                        padding='same',
                                        name='Maxpool_{}'.format(i+1))
      current_input = maxpool

    # Decode
    for i in range(self.layers_number):
      sample_up = tf.image.resize_images(current_input,
                                         size=(sizes[self.layers_number-i-1][1], sizes[self.layers_number-i-1][2]),
                                         method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
      conv = tf.layers.conv2d(inputs=sample_up,
                              filters=self.filters_number[self.layers_number-i-1],
                              kernel_size=self.filters_size[self.layers_number-i-1],
                              padding='same',
                              activation=tf.nn.relu,
                              name='Deconv_{}'.format(i+1))
      current_input = conv

    logit = tf.layers.conv2d(inputs=conv, filters=1, kernel_size=(3, 3), padding='same', activation=None)
    self.output = tf.nn.sigmoid(logit)

    loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=self.target, logits=logit)

    # Get cost and define the optimizer
    self.cost = tf.reduce_mean(loss)
    self.opt = tf.train.AdamOptimizer().minimize(self.cost)

  def init_session(self):
    self.sess = tf.Session()
    self.sess.run(tf.global_variables_initializer())

  def warmup(self):
    self.sess.run([self.output], feed_dict={self.input: np.zeros([1, self.input_size[0], self.input_size[1], 1])})

  def train(self, input_batch, target_batch):
    self.sess.run([self.opt], feed_dict={self.input: input_batch, self.target: target_batch})

  def infer(self, input):
    return self.sess.run([self.output], feed_dict={self.input: input})

if __name__ == '__main__':
  configs = {
    "input_size": (28, 28),
    "filters_number": [32, 32, 16],
    "filters_size": [(5, 5) for _ in range(3)]
  }

  model = AutoEncoder(configs)