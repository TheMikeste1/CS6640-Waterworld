import torch.nn
from pettingzoo.sisl import waterworld_v4 as waterworld

import custom_waterworld
from agents import NeuralNetwork, QNNAgent
from custom_waterworld import WaterworldArguments


def main():
    args = WaterworldArguments(
        FPS=60,
        render_mode=WaterworldArguments.RenderMode.NONE,
        max_cycles=512,
    )
    env = waterworld.env(**args.to_dict())

    num_obs = env.observation_space(env.possible_agents[0]).shape[0]

    # Create agents
    network = NeuralNetwork(
        layers=[
            torch.nn.Linear(num_obs, num_obs**2),
            torch.nn.ReLU(),
            torch.nn.Linear(num_obs**2, 64),
        ],
        optimizer_factory=torch.optim.Adam,
        optimizer_kwargs={"lr": 0.001},
        criterion_factory=torch.nn.MSELoss,
        criterion_kwargs={},
        lr_scheduler_factory=torch.optim.lr_scheduler.StepLR,
        lr_scheduler_kwargs={"step_size": 1, "gamma": 0.99},
    )

    policy_networks = [
        NeuralNetwork(
            layers=[
                torch.nn.Linear(64, 64),
                torch.nn.ReLU(),
                torch.nn.Linear(64, 10),
            ],
            optimizer_factory=torch.optim.Adam,
            optimizer_kwargs={"lr": 0.001},
            criterion_factory=torch.nn.MSELoss,
            criterion_kwargs={},
            lr_scheduler_factory=torch.optim.lr_scheduler.StepLR,
            lr_scheduler_kwargs={"step_size": 1, "gamma": 0.99},
        )
        for _ in range(2)
    ]

    agent = QNNAgent(
        env,
        "pursuer_0",
        value_model=network,
        policy_models=policy_networks,
    )

    runner = custom_waterworld.Runner(
        env,
        agents={
            "pursuer_0": agent,
            "pursuer_1": agent,
        },
        should_render_empty=args.render_mode == WaterworldArguments.RenderMode.HUMAN,
    )
    if args.render_mode == WaterworldArguments.RenderMode.HUMAN:
        print(f"Running at {env.unwrapped.env.FPS} FPS on {agent.device}")
    else:
        print(f"Running in the background on {agent.device}")

    try:
        runner.run_iterations(128)
    except KeyboardInterrupt:
        print("Run interrupted")
        return
    finally:
        env.close()

    env.unwrapped.env.render_mode = WaterworldArguments.RenderMode.RGB.value

    width, height = env.unwrapped.env.pixel_scale, env.unwrapped.env.pixel_scale
    vw = custom_waterworld.VideoWriter(env.unwrapped.env.FPS, width, height, "test.mp4")
    runner.on_render += lambda x, y: vw.write(y)
    runner.on_post_episode += lambda *_: vw.close()
    try:
        runner.run_episode()
    except KeyboardInterrupt:
        print("Run interrupted")
    finally:
        env.close()
        vw.close()


if __name__ == "__main__":
    main()
