import argparse

def read():
  parser = argparse.ArgumentParser()
  parser.add_argument("--max_episodes", type=int, default=20000,
                      help="Max number of times a game is run to learn")
  parser.add_argument("--max_epochs", type=int, default=300,
                      help="Max number of epochs per game")
  parser.add_argument("--discrete_buckets", type=int, default=10,
                      help="Buckets used to discretise the action space per dimension")
  parser.add_argument("--action_min", type=int, default=-1,
                      help="Minimum for action")
  parser.add_argument("--action_max", type=int, default=1,
                      help="Maximum for action")
  parser.add_argument("--batch_size", type=int, default=10,
                      help="Size of batch taken from experience")
  parser.add_argument("--experience_memory", type=int, default=500,
                      help="Memory of the experience")
  parser.add_argument("--gamma_parameter", type=float, default=0.8,
                      help="Gamma in the learning algorithm")
  parser.add_argument("--exploration_probability", type=float, default=0.1,
                      help="Probability of random action")
  parser.add_argument("--test_network", type=bool, default=False,
                      help="Just test the network with random parameters")
  parser.add_argument("--environment_name", type=str, default='BipedalWalker-v2',
                      help="Open gym environment used")
  return parser.parse_args()
