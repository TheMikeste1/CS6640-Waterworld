import pettingzoo.sisl.waterworld_v4 as waterworld
from pettingzoo.utils.conversions import parallel_wrapper_fn

#
# @property
# def base_env(self) -> waterworld.env:
#     return self.unwrapped.env
#
#
# def augmented_env(**kwargs):
#     env_ = waterworld.env(**kwargs)
#     setattr(env_.__class__, "base_env", base_env)
#     return env_
#
#
# env = augmented_env
#
#
# def augmented_parallel_env(**kwargs):
#     p_env = parallel_wrapper_fn(env)
#     env_ = p_env(**kwargs)
#     setattr(env_.__class__, "base_env", base_env)
#     return env_
#
#
# parallel_env = augmented_parallel_env
