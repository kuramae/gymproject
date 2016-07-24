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



class Agent:
  def __init__(self, options, state_dimensions, action_dimensions):
    self.action_dimensions = action_dimensions
    self.state_dimensions = state_dimensions
    self.options = options
    # TODO derive this based on the number of actions
    self.exploration_probability = options.exploration_probability
    self.batch_size = options.batch_size
    self.gamma_parameter = options.gamma_parameter
    self.experience_memory = options.experience_memory
    self.experience = []
    self.optimizer = tf.train.AdamOptimizer(learning_rate=0.01)

  def initialize(self):
    self.training_steps = 0
    self.session = tf.Session()
    self.network = self.build_inner_network()
    self.define_placeholders()
    self.define_variables(self.network)
    self.train_writer = tf.train.SummaryWriter('/tmp/train',
                                          self.session.graph)
    self.merged = tf.merge_all_summaries()
    self.session.run(tf.initialize_all_variables())

  def store(self, observation, action, reward, newobservation):
    if len(self.experience) > self.experience_memory:
      self.experience.pop(0)
    self.experience.append((observation, action, reward, newobservation))


  def define_placeholders(self):
    """This is what comes in the network
    """
    # Given a current state, we find the best action to perform
    # The None part means it's variable (it will be part of the batch)
    # So the current state will be a matrix of the type
    # a b c d
    # e f g h
    # i j k l
    # (if the batch has size 3 and the state has size 4)
    self.observation = tf.placeholder("float", [None, self.state_dimensions], name="observation")
    self.next_observation = tf.placeholder(tf.float32, self.state_batch_shape(), name="next_observation")
    # TODO: why stop gradient?
    # self.next_action_scores = tf.stop_gradient(self.network(self.next_observation))
    self.next_action_scores = tf.stop_gradient(self.network(self.next_observation))
    self.rewards = tf.placeholder(tf.float32, (None,), name="rewards")
    tf.histogram_summary("rewards", self.rewards)
    # Could set batch_size rather than None but this leaves it more flexible
    self.next_observation_mask = tf.placeholder(tf.float32, (None,), name="next_observation_mask")
    self.action_mask = tf.placeholder(tf.float32, (None, self.action_dimensions), name="action_mask")

  def define_variables(self, network):
    # These are used to go from an observation to an action
    with tf.name_scope("action"):
      # Puts the observation in, and it gets the scores out
      self.action_scores = tf.identity(network(self.observation), name="action_scores")
      tf.histogram_summary("action_scores", self.action_scores)
      # Returns the index of the action with the highest action_score
      self.predicted_actions = tf.argmax(self.action_scores, dimension=1, name="predicted_actions")
      tf.histogram_summary("predicted_actions", self.predicted_actions)

    # This is to predict future rewards based on a batch of experiences (it's what we sum to the reward)
    with tf.name_scope("estimating_future_rewards"):
      max_next_actions = tf.reduce_max(self.next_action_scores, reduction_indices=[1, ]) * self.next_observation_mask
      # There's a reward and an action for each action
      self.future_rewards = self.rewards + self.gamma_parameter * max_next_actions
      tf.histogram_summary("future_rewards", self.future_rewards)

    # This is to get the full q-value
    with tf.name_scope("q_value"):
      self.masked_action_scores = tf.reduce_sum(self.action_scores * self.action_mask, reduction_indices=[1, ])
      tf.histogram_summary("masked_action_scores", self.masked_action_scores)
      temp_diff = self.masked_action_scores - self.future_rewards
      self.prediction_error = tf.reduce_mean(tf.square(temp_diff))
      tf.histogram_summary("prediction_error", self.prediction_error)
      gradients = self.optimizer.compute_gradients(self.prediction_error)
      for i, (grad, var) in enumerate(gradients):
        if grad is not None:
          gradients[i] = (tf.clip_by_norm(grad, 5), var)
      # Add histograms for gradients.
      # for grad, var in gradients:
      #   tf.histogram_summary(var.name, var)
      #   if grad is not None:
      #     tf.histogram_summary(var.name + '/gradients', grad)
      self.train_op = self.optimizer.apply_gradients(gradients)

  def build_inner_network(self):
    # First layer input: states, output: 200 (input layer)
    # Second layer input: 200, output: 200 (hidden layer 0)
    # Second layer input: 200, output: num_actions (hidden layer 2)
    return MLP([self.state_dimensions, ], [HIDDEN_NEURONS, HIDDEN_NEURONS, self.action_dimensions],
               [tf.tanh, tf.tanh, tf.identity])

  def state_batch_shape(self):
    return (self.batch_size, self.state_dimensions)

  def training_step(self):
    """Pick a self.minibatch_size exeperiences from reply buffer
    and backpropage the value function.
    """
    if len(self.experience) < self.batch_size:
      return

    # sample experience.
    samples = random.sample(self.experience, self.batch_size)

    # bach states
    # if we have 2 state dimensions and a batch of five we have a 5x2 matrix [[a,b], [c,d], ...]
    states = np.empty(self.state_batch_shape())
    new_states = np.empty(self.state_batch_shape())
    action_mask = np.zeros((self.batch_size, self.action_dimensions))
    new_states_mask = np.empty((self.batch_size,))
    # One reward for each batch element (and for each action)
    rewards = np.empty((self.batch_size,))

    for i, (state, action, reward, newstate) in enumerate(samples):
      states[i] = state
      action_mask[i] = 0
      action_mask[i][action] = 1
      rewards[i] = reward
      if newstate is not None:
        new_states[i] = newstate
        new_states_mask[i] = 1
      else:
        new_states[i] = 0
        new_states_mask[i] = 0

    summary = self.session.run([
      self.merged
    ], {
      self.observation: states,
      self.next_observation: new_states,
      self.next_observation_mask: new_states_mask,
      self.action_mask: action_mask,
      self.rewards: rewards
    })
    self.train_writer.add_summary(summary[0], self.training_steps)
    self.training_steps += 1

  def run_action(self, observation):
    """Given observation returns the action that should be chosen using
    DeepQ learning strategy. Does not backprop."""
    assert len(observation) == self.state_dimensions, "dimensions differ %s != %s" % (len(observation) , self.state_dimensions)
    if random.random() < self.exploration_probability:
      return random.randint(0, self.action_dimensions - 1)
    else:
      logger.debug("Running with observation %s", observation)
      # [observation] is like [[a,b,c]]
      return self.session.run(self.predicted_actions, {self.observation: [observation]})[0]


class MLP(object):
  def __init__(self, input_sizes, hiddens, nonlinearities):
    self.input_sizes = input_sizes
    self.hiddens = hiddens
    self.input_nonlinearity, self.layer_nonlinearities = nonlinearities[0], nonlinearities[1:]
    self.scope = "brain"

    assert len(hiddens) == len(nonlinearities), \
      "Number of hiddens must be equal to number of nonlinearities"

    with tf.variable_scope(self.scope):
      self.input_layer = Layer(input_sizes, hiddens[0], scope="input_layer")
      self.layers = []
      for l_idx, (h_from, h_to) in enumerate(zip(hiddens[:-1], hiddens[1:])):
        self.layers.append(Layer(h_from, h_to, scope="hidden_layer_%d" % (l_idx,)))

  def __call__(self, xs):
    if type(xs) != list:
      xs = [xs]
    with tf.variable_scope(self.scope):
      hidden = self.input_nonlinearity(self.input_layer(xs))
      for layer, nonlinearity in zip(self.layers, self.layer_nonlinearities):
        hidden = nonlinearity(layer(hidden))
      return hidden

  def variables(self):
    res = self.input_layer.variables()
    for layer in self.layers:
      res.extend(layer.variables())
    return res


class Layer(object):
  def __init__(self, input_sizes, output_size, scope):
    """Cretes a neural network layer."""
    if type(input_sizes) != list:
      input_sizes = [input_sizes]

    self.input_sizes = input_sizes
    self.output_size = output_size
    self.scope = scope or "Layer"

    with tf.variable_scope(self.scope):
      self.Ws = []
      for input_idx, input_size in enumerate(input_sizes):
        W_name = "W_%d" % (input_idx,)
        W_initializer = tf.random_uniform_initializer(
          -1.0 / math.sqrt(input_size), 1.0 / math.sqrt(input_size))
        W_var = tf.get_variable(W_name, (input_size, output_size), initializer=W_initializer)
        self.Ws.append(W_var)
      self.b = tf.get_variable("b", (output_size,), initializer=tf.constant_initializer(0))

  def __call__(self, xs):
    if type(xs) != list:
      xs = [xs]
    assert len(xs) == len(self.Ws), \
      "Expected %d input vectors, got %d" % (len(self.Ws), len(xs))
    with tf.variable_scope(self.scope):
      return sum([tf.matmul(x, W) for x, W in zip(xs, self.Ws)]) + self.b

  def variables(self):
    return [self.b] + self.Ws