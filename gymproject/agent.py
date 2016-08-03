import tensorflow as tf
import numpy as np
import math
import random
import logging

HIDDEN_NEURONS = 200

logger = logging.getLogger(__name__)

class ContinuousAgent:
  def __init__(self, options, state_dimensions, action_dimensions):
    self.discrete_buckets = options.discrete_buckets
    self.action_min = options.action_min
    self.action_max = options.action_max
    self.action_dimensions = action_dimensions
    self.quant = (self.action_max - self.action_min) / float(self.discrete_buckets)
    self.agent = Agent(options, int(state_dimensions), self.num_discrete_actions(options.discrete_buckets, int(action_dimensions)))

  def initialize(self):
    self.agent.initialize()

  def number_to_quant(self, number):
    return self.action_min + self.quant * number

  def num_discrete_actions(self, discrete_buckets, action_dimensions):
    return int(math.pow(discrete_buckets, action_dimensions))

  def store(self, observation, action, reward, new_observation):
    self.agent.store(observation, self.map_continuous_to_discrete(action), reward, new_observation)

  def map_continuous_to_discrete(self, action):
    # Action is of the type [a, b, c, d]
    # from that we return a scalar
    multiplier = 0
    total = 0
    for dimension in action:
      # if min = -1 and max =1 and we have 10 buckets then a quant is 0.2
      # if a quant is 0.2 and dimension is 0.7 discrete dimension is 1.7/0.2 = 8
      if dimension == self.action_max:
        # we don't want to overflow the buckets, if we have 10 of them, the max is 9
        discrete_dimension = self.discrete_buckets - 1
      else:
        discrete_dimension = math.floor((dimension - self.action_min) / self.quant)
      # if 12 buckets [0, 0, 0] -> 0, [0, 0, 11] -> 11, [0, 1, 4] -> 16, [2, 2, 2] -> 2 + 12 * 2 + 12*12 *2
      total+= math.pow(self.discrete_buckets, multiplier) * discrete_dimension
      multiplier+=1
    return total

  def map_discrete_to_continuous(self, action):
    output = [0] * self.action_dimensions
    current_action = action
    # Suppose action_dimensions is 4 and action is 134 and buckets are x + y * 5  + z * 25 + w * 125 = 134
    for dimension in range(self.action_dimensions - 1, -1, -1):
      pow = math.pow(self.discrete_buckets, dimension)
      number = math.floor(current_action / pow)
      output[dimension] = self.number_to_quant(number)
      current_action = current_action % pow
    return output

  def act(self, observation):
    # Observation is of the type [[i, j, k, l]] which is a row in a 1xD matrix (rows are used for batch observations)
    predicted_discrete_action = self.agent.run_action(observation)
    return self.map_discrete_to_continuous(predicted_discrete_action)

  def train(self):
    self.agent.training_step()


# The training is made of experiences of the type (observation, action, reward, newobservation)
# reward is r, observation is s and new observation is s', action is a
# Q(s, a) = r + gamma * max_a'(Q(s', a'))
# Q(s, a) is partially obseved, so r, a, s' and s all come from the exerpience
# but we use the network to get max_a'(Q(s', a')).
# We get max_a'(Q(s', a')) and then we use r + gamma * max_a'(Q(s', a')) as
# y of the network (expected value). The loss that we give as input is is y - Q(s, a).
class Agent:
  def __init__(self, options, state_dimensions, action_dimensions):
    self.num_possible_actions = action_dimensions
    self.state_dimensions = state_dimensions
    # TODO derive this based on the number of actions
    self.exploration_probability = options.exploration_probability
    self.batch_size = options.batch_size
    self.gamma_parameter = options.gamma_parameter
    self.experience_memory = options.experience_memory


  def initialize(self):
    self.training_steps = 0
    self._experience = []
    self._session = tf.Session()
    self._network = self.build_inner_network()
    self._network_input = self._network.get_input()
    self._network_output = self._network.get_output()
    self._target = tf.placeholder("float", [None])
    # It picks the action network output is a (batch X action_dimensions) matrix and self.action is
    # a (action_dimensions x action_dimensions) matrix where each row is a [0 ... 1 ... 0] type vector
    # By multiplying (element-wise) we get a (batch x action_dimension) and by summing we get a vector with all the
    # rewards for each element of the batch
    self._picked_action_vector = tf.placeholder("float", [self.batch_size, self.num_possible_actions], name="picked_action_vector")
    q_of_current_state = tf.reduce_sum(tf.mul(self._network_output, self._picked_action_vector, name="q_of_current_state"), reduction_indices=1)
    self._cost = tf.reduce_mean(tf.square(self._target - q_of_current_state), name="cost")
    self._train_operation = tf.train.AdamOptimizer(1e-2).minimize(self._cost)
    self._define_histograms()
    self._train_writer = tf.train.SummaryWriter('/tmp/train', self._session.graph)
    self._merged = tf.merge_all_summaries()
    self._session.run(tf.initialize_all_variables())

  def store(self, observation, action, reward, newobservation):
    if len(self._experience) > self.experience_memory:
      self._experience.pop(0)
    self._experience.append((observation, int(action), reward, newobservation))

  def _action_index_to_vector(self, index):
    vector = [0] * self.num_possible_actions
    # TODO ew
    vector[index.tolist()[0]] = 1
    return vector

  def _define_histograms(self):
    tf.histogram_summary("cost", self._cost)

  def build_inner_network(self):
    # First layer input: states, output: 200 (input layer)
    # Second layer input: 200, output: 200 (hidden layer 0)
    # Second layer input: 200, output: num_actions (hidden layer 2)
    return MLP(self.state_dimensions, self.num_possible_actions)

  def state_batch_shape(self):
    return (self.batch_size, self.state_dimensions)

  def action_batch_shape(self):
    return (self.batch_size, self.num_possible_actions)

  def training_step(self):
    """Pick a self.minibatch_size exeperiences from reply buffer
    and backpropage the value function.
    """
    if len(self._experience) < self.batch_size:
      return

    # sample experience.
    samples = random.sample(self._experience, self.batch_size)

    # bach states
    # if we have 2 state dimensions and a batch of five we have a 5x2 matrix [[a,b], [c,d], ...]
    states = np.empty(self.state_batch_shape())
    new_states = np.empty(self.state_batch_shape())
    actions = np.empty((self.batch_size, 1), dtype=int)
    # One reward for each batch element (and for each action)
    rewards = np.empty((self.batch_size,))

    for i, (state, action, reward, newstate) in enumerate(samples):
      states[i] = state
      rewards[i] = reward
      new_states[i] = newstate
      # this is just the index of the Q function
      actions[i] = action

    # Gets the q function value for each action (1xaction_dimension) vector
    q_function_per_action = self._network_output.eval(session=self._session, feed_dict={self._network_input: self.matrix_to_tensor(new_states)})
    expected_q = []
    for i in range(len(samples)):
      expected_q.append(
          rewards[i] + self.gamma_parameter * np.max(q_function_per_action[i]))

    state_tensor = self.matrix_to_tensor(states)
    action_tensor = [self._action_index_to_vector(a) for a in actions]
    summary = self._train_operation.run(
      session=self._session,
      feed_dict={
      self._network_input: state_tensor,
      self._picked_action_vector: action_tensor,
      self._target: expected_q})


    #self.train_writer.add_summary(summary[0], self.training_steps)
    self.training_steps += 1

  def run_action(self, observation):
    """Given observation returns the action that should be chosen using
    DeepQ learning strategy. Does not backprop."""
    assert len(observation) == self.state_dimensions, "dimensions differ %s != %s" % \
                                                      (len(observation), self.state_dimensions)
    if random.random() < self.exploration_probability:
      return_action_index = random.randint(0, self.num_possible_actions - 1)
    else:
      q_function_per_action = self._session.run(self._network_output, feed_dict={self._network_input: self.vector_to_tensor(observation)})[0]
      action_index = np.argmax(q_function_per_action)
      return_action_index = action_index
    assert return_action_index < self.num_possible_actions and return_action_index >= 0, "action index outside boundary %s" % \
                                                                                         (return_action_index)
    return return_action_index

  # from (X, Y) to (X, 1, Y, 1)
  def matrix_to_tensor(self, matrix):
    batch_rows = []
    for b in matrix:
      batch_rows.append([[[i] for i in b]])
    return batch_rows

  # from (X) to (1, 1, Y, 1)
  def vector_to_tensor(self, vector):
    return [[[[i] for i in vector]]]



class MLP(object):
  def __init__(self, input_size, output_size):
    self.scope = "brain"

    with tf.variable_scope(self.scope):
      # We keep height to 1
      HEIGHT = 1
      # Batch dimension is flexible
      BATCH_DIMENSION = None
      INPUT_LAYER_DEPTH = 1
      PADDING = "SAME"
      input_layer = tf.placeholder("float", [BATCH_DIMENSION, HEIGHT, input_size, INPUT_LAYER_DEPTH])

      # First Layer
      convolution_weights_1 = tf.Variable(tf.truncated_normal([HEIGHT, 4, 1, 32], stddev=0.01))
      convolution_bias_1 = tf.Variable(tf.constant(0.01, shape=[32]))
      hidden_convolutional_layer_1 = tf.nn.relu(
        tf.nn.conv2d(input_layer, convolution_weights_1, strides=[HEIGHT, 2, 1, 1], padding=PADDING) + convolution_bias_1)

      # Second Layer
      convolution_weights_2 = tf.Variable(tf.truncated_normal([HEIGHT, 4, 32, 64], stddev=0.01))
      convolution_bias_2 = tf.Variable(tf.constant(0.01, shape=[64]))
      hidden_convolutional_layer_2 = tf.nn.relu(
        tf.nn.conv2d(hidden_convolutional_layer_1, convolution_weights_2, strides=[1, 2, 1, 1],
                     padding=PADDING) + convolution_bias_2)

      # Third layer
      convolution_weights_3 = tf.Variable(tf.truncated_normal([HEIGHT, 4, 64, 64], stddev=0.01))
      convolution_bias_3 = tf.Variable(tf.constant(0.01, shape=[64]))
      hidden_convolutional_layer_3 = tf.nn.relu(
        tf.nn.conv2d(hidden_convolutional_layer_2, convolution_weights_3,
                     strides=[1, 2, 1, 1], padding=PADDING) + convolution_bias_3)
      # TODO works only for 24 states
      magic_number = 6
      hidden_convolutional_layer_3_flat = tf.reshape(hidden_convolutional_layer_3, [-1, 256 * magic_number])

      # Fourth Layer
      feed_forward_weights_1 = tf.Variable(tf.truncated_normal([256 * magic_number, 256], stddev=0.01))
      feed_forward_bias_1 = tf.Variable(tf.constant(0.01, shape=[256]))
      final_hidden_activations = tf.nn.relu(
        tf.matmul(hidden_convolutional_layer_3_flat, feed_forward_weights_1) + feed_forward_bias_1)

      # Output Layer
      feed_forward_weights_2 = tf.Variable(tf.truncated_normal([256, output_size], stddev=0.01))
      feed_forward_bias_2 = tf.Variable(tf.constant(0.01, shape=[output_size]))
      output_layer = tf.matmul(final_hidden_activations, feed_forward_weights_2) + feed_forward_bias_2

      self.input_layer = input_layer
      # learn that these actions in these states lead to this reward
      self.output_layer = output_layer

  def get_input(self):
    return self.input_layer

  def get_output(self):
    return self.output_layer
