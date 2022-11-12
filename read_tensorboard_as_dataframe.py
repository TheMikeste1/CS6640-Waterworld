import pandas as pd
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


def read_tensorboard_as_dataframe(path: str):
    # https://github.com/theRealSuperMario/supermariopy/blob/master/scripts/tflogs2pandas.py#L14
    df = pd.DataFrame()
    acc = EventAccumulator(path, size_guidance={"scalars": 0})

    acc.Reload()
    tags = acc.Tags()["scalars"]
    for tag in tags:
        event_list = acc.Scalars(tag)
        values = list(map(lambda x: x.value, event_list))
        step = list(map(lambda x: x.step, event_list))
        r = {"metric": [tag] * len(step), "value": values, "step": step}
        r = pd.DataFrame(r)
        df = pd.concat([df, r])

    return df
