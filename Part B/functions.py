import pandas as pd

SENSOR_LIST = ["x1", "y1", "z1", "x2", "y2", "z2", "x3", "y3", "z3", "x4", "y4", "z4"]


def getClassStats(df: pd.DataFrame) -> pd.DataFrame:
    classes = df["class"].unique()

    rows = []
    for cl in classes:
        class_data_df = df.loc[df["class"] == cl].drop("class", axis=1)
        rows.append(class_data_df.mean().to_list() + [cl, "Average"])
        rows.append(class_data_df.min().to_list() + [cl, "Min"])
        rows.append(class_data_df.max().to_list() + [cl, "Max"])

    return pd.DataFrame(rows, columns=SENSOR_LIST + ["class", "metric"], index=None)
