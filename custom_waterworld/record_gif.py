# import imageio
#
#
# def run_episode(env: pz.AECEnv, agents, memory):
#     rewards = defaultdict(list)
#     env.reset()
#     env.render()
#     with imageio.get_writer('test.gif', mode='I', fps=env.base_env.FPS) as writer:
#         for agent in env.agent_iter():
#             obs, reward, terminated, truncated, info = env.last()
#             rewards[agent].append(reward)
#             action = None if terminated or truncated else env.action_space(agent).sample()
#             env.step(action)
#             out = env.render()
#             writer.append_data(out)
