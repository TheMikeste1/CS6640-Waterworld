from functools import partial

import torch.nn
import torchinfo
from pettingzoo.sisl import waterworld_v4 as waterworld
from torch.utils.tensorboard import SummaryWriter

import custom_waterworld
from agents import NeuralNetwork, QNNAgent
from custom_waterworld import WaterworldArguments


def on_post_episode(writer, runner, it, rewards, agent_posts, agent_trains):
    for agent_name in agent_posts.keys():
        reward = rewards[agent_name]
        writer.add_scalar(f"{agent_name}/reward", sum(reward), it)


def on_post_train(writer, runner, it, agent_trains):
    for agent_name in agent_trains.keys():
        loss = agent_trains[agent_name]
        writer.add_scalar(f"{agent_name}/loss", loss, it)


def main():
    args = WaterworldArguments(
        FPS=60,
        render_mode=WaterworldArguments.RenderMode.NONE,
        max_cycles=512,
        n_evaders=20 * 3,
        n_poisons=20 * 3,
    )
    env = waterworld.env(**args.to_dict())

    num_obs = env.observation_space(env.possible_agents[0]).shape[0]

    # Create agents
    policy_networks = [
        NeuralNetwork(
            layers=[
                torch.nn.Linear(num_obs, 256),
                torch.nn.ReLU(),
                torch.nn.Linear(256, 64),
                torch.nn.ReLU(),
                torch.nn.Linear(64, 3),
            ],
            optimizer_factory=torch.optim.Adam,
            optimizer_kwargs={"lr": 0.001},
            criterion_factory=torch.nn.SmoothL1Loss,
            criterion_kwargs={},
            lr_scheduler_factory=torch.optim.lr_scheduler.StepLR,
            lr_scheduler_kwargs={"step_size": 1, "gamma": 0.99},
        )
        for _ in range(2)
    ]

    pursuer_0 = QNNAgent(
        env,
        "pursuer_0",
        policy_models=policy_networks,
        batch_size=512,
    )
    pursuer_0.enable_explore = False
    torchinfo.summary(
        pursuer_0, input_size=(1, num_obs), device=pursuer_0.device, depth=5
    )
    pursuer_0.enable_explore = True

    pursuer_1 = QNNAgent(
        env,
        "pursuer_1",
        policy_models=policy_networks,
        batch_size=512,
    )

    runner = custom_waterworld.Runner(
        env,
        agents={
            "pursuer_0": pursuer_0,
            "pursuer_1": pursuer_1,
        },
        should_render_empty=args.render_mode == WaterworldArguments.RenderMode.HUMAN,
    )

    tensorboard_writer = SummaryWriter()
    write_post_episode = partial(on_post_episode, tensorboard_writer)
    runner.on_post_episode += write_post_episode

    if args.render_mode == WaterworldArguments.RenderMode.HUMAN:
        print(f"Running at {env.unwrapped.env.FPS} FPS on {pursuer_0.device}")
    else:
        print(f"Running in the background on {pursuer_0.device}")
    print("", end="", flush=True)

    # Train
    try:
        runner.run_iterations(128)
    except KeyboardInterrupt:
        print("Run interrupted")
        return
    finally:
        env.close()
        tensorboard_writer.close()
        torch.save(pursuer_0.state_dict(), f"agent.pt")

    # Run an episode to film
    env.unwrapped.env.render_mode = WaterworldArguments.RenderMode.RGB.value

    width, height = env.unwrapped.env.pixel_scale, env.unwrapped.env.pixel_scale
    visual_writer = custom_waterworld.VideoWriter(
        env.unwrapped.env.FPS, width, height, "test.mp4"
    )
    # visual_writer = custom_waterworld.GIFWriter(env.unwrapped.env.FPS, "test.gif")
    runner.on_render += lambda x, y: visual_writer.write(y)
    runner.on_post_episode += lambda *_: visual_writer.close()

    pursuer_0.enable_explore = False
    pursuer_1.enable_explore = False
    try:
        runner.run_episode(train=False)
    except KeyboardInterrupt:
        print("Run interrupted")
    finally:
        env.close()
        tensorboard_writer.close()
        visual_writer.close()


if __name__ == "__main__":
    main()
