import numpy as np
import pandas as pd

from src.config import observations_dir


def data2float(v):
    try:
        return float(v)
    except:
        return np.nan


def process_windGuru_data(filename):
    data = pd.read_csv(f"{observations_dir}/{filename}.csv").melt(id_vars="Unnamed: 0")
    data.rename({"Unnamed: 0": "Date", "variable": "Hour", "value": filename}, axis=1, inplace=True)
    data["Date"] = pd.to_datetime(data.apply(lambda df: f"{df['Date']}/{df['Hour'][:-1]}", axis=1),
                                  dayfirst=True)
    data.drop("Hour", axis=1, inplace=True)
    data.set_index("Date", inplace=True)
    data.sort_index(inplace=True)
    return data.applymap(data2float)
