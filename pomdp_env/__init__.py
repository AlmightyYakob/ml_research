import logging
from gym.envs.registration import register

logger = logging.getLogger(__name__)

register(
    id="POMDP-v0",
    entry_point="pomdp_env.envs:POMDPEnv",
    # timestep_limit=1000,
    # reward_threshold=1.0,
    nondeterministic=True,
)
