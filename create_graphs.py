from matplotlib import pyplot as plt
import pandas as pd
import seaborn as sns

from read_tensorboard_as_dataframe import read_tensorboard_as_dataframe

pd.options.mode.chained_assignment = None  # default='warn'

# %% Pre-fix
df = read_tensorboard_as_dataframe(
    "runs/keep/2022-11-09_20-03-02_qnn_distance_12288its/"
    "events.out.tfevents.1668049382.TheMikeste1.22600.0"
)

df = df[df["step"] < 4096]

df["agent"] = df["metric"].apply(lambda s: s.split("/")[0])
df["metric"] = df["metric"].apply(lambda s: s.split("/")[1])

df_plot = df[(df["metric"] == "total_reward")]

agents = df_plot["agent"].unique()
for agent in agents:
    df_plot.loc[df_plot["agent"] == agent, "value"] = (
        df_plot.loc[df_plot["agent"] == agent, "value"].ewm(alpha=(1 - 0.999)).mean()
    )

sns.set_style("whitegrid")
plot = sns.lineplot(data=df_plot, x="step", y="value", hue="agent")
plot.set(xlabel="Step", ylabel="Total Reward")
plot.title.set_text("Total Reward of Agents during Training: Pre-fix")
plot.get_figure().savefig("total_reward_pre_fix.png")
plot.get_figure().show()

# Loss
df_plot = df[(df["metric"] == "loss")]
# Plot original loss transparently
df_plot["_agent"] = df_plot["agent"].apply(lambda s: "_" + s)
plot = sns.lineplot(data=df_plot, x="step", y="value", hue="_agent", alpha=0.25)

agents = df_plot["agent"].unique()
for agent in agents:
    df_plot.loc[df_plot["agent"] == agent, "value"] = (
        df_plot.loc[df_plot["agent"] == agent, "value"].ewm(alpha=(1 - 0.999)).mean()
    )

sns.set_style("whitegrid")
plot = sns.lineplot(data=df_plot, x="step", y="value", hue="agent", ax=plot)
plot.set(xlabel="Step", ylabel="Loss (Cross Entropy)")
plot.title.set_text("Loss during Training: Pre-fix")
plot.get_figure().savefig("loss_pre_fix.png")
plot.get_figure().show()


# %% Post fix
df = read_tensorboard_as_dataframe(
    "runs/keep/qnn_distance_4096its_2022-11-10_17-14-43/"
    "events.out.tfevents.1668125683.TheMikeste1.12548.0"
)

df = df[df["step"] < 4096]

df["agent"] = df["metric"].apply(lambda s: s.split("/")[0])
df["metric"] = df["metric"].apply(lambda s: s.split("/")[1])

df_plot = df[(df["metric"] == "total_reward")]

agents = df_plot["agent"].unique()
for agent in agents:
    df_plot.loc[df_plot["agent"] == agent, "value"] = (
        df_plot.loc[df_plot["agent"] == agent, "value"].ewm(alpha=(1 - 0.999)).mean()
    )

sns.set_style("whitegrid")
plot = sns.lineplot(data=df_plot, x="step", y="value", hue="agent")
plot.set(xlabel="Step", ylabel="Total Reward")
plot.title.set_text("Total Reward of Agents during Training: Post-fix")
plot.get_figure().savefig("total_reward_post_fix.png")
plot.get_figure().show()

# Loss
df_plot = df[(df["metric"] == "loss")]
# Plot original loss transparently
df_plot["_agent"] = df_plot["agent"].apply(lambda s: "_" + s)
plot = sns.lineplot(data=df_plot, x="step", y="value", hue="_agent", alpha=0.25)

agents = df_plot["agent"].unique()
for agent in agents:
    df_plot.loc[df_plot["agent"] == agent, "value"] = (
        df_plot.loc[df_plot["agent"] == agent, "value"].ewm(alpha=(1 - 0.999)).mean()
    )

sns.set_style("whitegrid")
plot = sns.lineplot(data=df_plot, x="step", y="value", hue="agent", ax=plot)
plot.set(xlabel="Step", ylabel="Loss (Cross Entropy)")
plot.title.set_text("Loss during Training: Post-fix")
plot.get_figure().savefig("loss_post_fix.png")
plot.get_figure().show()
