from .env import Env
from .recorder import Recorder

try:
  import gym
  gym.register(
      id='HomeostaticCrafter-v1',
      entry_point='homeostatic_crafter:Env',
      max_episode_steps=10000,
      kwargs={'reward': True, 'random_internal': True, 'render_mode': None, 'homeostatic': True})
  # gym.register(
  #     id='CrafterNoReward-v1',
  #     entry_point='crafter:Env',
  #     max_episode_steps=10000,
  #     kwargs={'reward': False})
except ImportError:
  pass
