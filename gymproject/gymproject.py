from __future__ import print_function
import numpy as np
import logging
import gym

import config
from agent import ContinuousAgent

DEBUG = 10

INFO = 20

TEST_SPACE_ARRAY = [-1.39931729e-02, 9.64386845e-02, 3.19248247e-02, 2.79749084e-02, -3.51527303e-01, -3.38940233e-01,
                    9.09134746e-02, -1.00000032e+00, 0.00000000e+00, 8.60869765e-01, -1.00000918e+00, -6.41238213e-01,
                    -7.49031703e-06, 1.00000000e+00, 3.54064763e-01, 3.58085692e-01, 3.70617867e-01, 3.93209994e-01,
                    4.28994894e-01, 4.83900726e-01, 5.69593132e-01, 7.11586356e-01, 9.77015913e-01, 1.00000000e+00]

logging.basicConfig(format='%(asctime)s %(message)s')
logger = logging.getLogger(__name__)
logger.setLevel(DEBUG)


def main():
  options = config.read()
  logger.info("Running with options %s", options)
  env = gym.make(options.environment_name)
  env.reset()

  # TODO: dimensions shouldn't be hard coded, they can come from the env
  agent = ContinuousAgent(options, 24, 4)
  agent.initialize()
  if options.test_network:
    logger.action_space("Action space: %s", env.action_space)
    logger.action_space("Observation space: %s", env.observation_space)
    action = agent.act(TEST_SPACE_ARRAY)
    print(action)
  else:
    run_training(agent, env, options)


def run_training(agent, env, options):
  for i_episode in range(options.max_episodes):
    state = env.reset()
    for t in range(options.max_epochs):
      env.render()
      # Act based on the state
      action_for_state = agent.act(state)
      new_state, reward, done, info = env.step(np.array(action_for_state, dtype=np.float32))
      # Store the learned experience
      agent.store(state, action_for_state, reward, new_state)
      # Run a round of batch learning
      agent.train()
      # Update state
      state = new_state
      if reward > 0.5:
        logger.debug("Reward: %s", reward)
      if done:
        logger.debug("Episode finished after {} timesteps".format(t + 1))
        break


if __name__ == "__main__":
  main()