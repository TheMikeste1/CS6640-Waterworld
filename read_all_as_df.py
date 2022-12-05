import os

import pandas as pd

from read_tensorboard_as_dataframe import read_tensorboard_as_dataframe


def read_all_as_df(path: str):
    df_all = pd.DataFrame()
    for root, _, files in os.walk(path):
        run_name = root.split(os.path.sep)[-1]
        for file in files:
            file_path = os.path.join(root, file)
            df = read_tensorboard_as_dataframe(file_path)
            df["run_name"] = run_name
            df_all = pd.concat([df_all, df])
    df_all.reset_index(inplace=True, drop=True)
    df_all["agent"] = df_all["metric"].str.split("/").str[0].astype("category")
    df_all["metric"] = df_all["metric"].str.split("/").str[1].astype("category")
    df_all["run_name"] = df_all["run_name"].astype("category")
    df_all.sort_index(axis=1, inplace=True)
    return df_all
