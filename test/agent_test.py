import unittest

import math

from gymproject.agent import ContinuousAgent



STATE_DIMENSIONS = 24
ACTION_DIMENSIONS = 3

class AttributeDict(dict):
  __getattr__ = dict.__getitem__
  __setattr__ = dict.__setitem__

DEFAULT_OPTIONS = AttributeDict()
DEFAULT_OPTIONS.discrete_buckets = 10
DEFAULT_OPTIONS.action_min = -1
DEFAULT_OPTIONS.action_max = 1
DEFAULT_OPTIONS.batch_size = 5
DEFAULT_OPTIONS.gamma_parameter = 0.1
DEFAULT_OPTIONS.exploration_probability = 0.1

class AgentTest(unittest.TestCase):
  def setUp(self):
    self.agent = ContinuousAgent(DEFAULT_OPTIONS, STATE_DIMENSIONS, ACTION_DIMENSIONS)

  def test_number_to_quant(self):
    self.assertEqual(self.agent.number_to_quant(0),
                     DEFAULT_OPTIONS.action_min)
    self.assertEqual(self.agent.number_to_quant(DEFAULT_OPTIONS.discrete_buckets),
                     DEFAULT_OPTIONS.action_max)
    self.assertEqual(self.agent.number_to_quant(DEFAULT_OPTIONS.discrete_buckets/2),
                     DEFAULT_OPTIONS.action_min + (DEFAULT_OPTIONS.action_max - DEFAULT_OPTIONS.action_min)/2)

  def test_map_discrete_to_continuous(self):
    self.assertEqual(self.agent.map_discrete_to_continuous(0),
                     [DEFAULT_OPTIONS.action_min, DEFAULT_OPTIONS.action_min, DEFAULT_OPTIONS.action_min])
    self.assertEqual(self.agent.map_discrete_to_continuous(1),
                     [-0.8, DEFAULT_OPTIONS.action_min, DEFAULT_OPTIONS.action_min])
    self.assertEqual(self.agent.map_discrete_to_continuous(10),
                     [DEFAULT_OPTIONS.action_min, -0.8, DEFAULT_OPTIONS.action_min])
    self.assertEqual(self.agent.map_discrete_to_continuous(9),
                     [0.8, DEFAULT_OPTIONS.action_min, DEFAULT_OPTIONS.action_min])

  def test_map_continuous_to_discrete(self):
    self.assertEqual(0,
                     self.agent.map_continuous_to_discrete([DEFAULT_OPTIONS.action_min, DEFAULT_OPTIONS.action_min, DEFAULT_OPTIONS.action_min]))
    self.assertEqual(1,
                     self.agent.map_continuous_to_discrete([-0.79, DEFAULT_OPTIONS.action_min, DEFAULT_OPTIONS.action_min]))
    self.assertEqual(10,
                     self.agent.map_continuous_to_discrete([DEFAULT_OPTIONS.action_min, -0.79, DEFAULT_OPTIONS.action_min]))
    self.assertEqual(9,
                     self.agent.map_continuous_to_discrete([0.81, DEFAULT_OPTIONS.action_min, DEFAULT_OPTIONS.action_min]))

  def test_map_discrete_to_continuous_all_different(self):
    discrete_action_number = self.agent.num_discrete_actions()
    observed = set()
    for i in range(discrete_action_number):
      action = tuple(self.agent.map_discrete_to_continuous(i))
      self.assertFalse(action in observed)
      observed.add(action)
    self.assertEquals(len(observed), discrete_action_number)
