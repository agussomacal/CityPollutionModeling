from datetime import datetime

import numpy as np

import pandas as pd

from src.config import observations_dir


def load_pollution_file(year):
    pollution = pd.read_csv(f"{observations_dir}/{year}_NO2.csv", index_col=0, header=2)[3:]
    pollution.index = pd.to_datetime(pollution.index).tz_localize(None)
    pollution = pollution.astype(float)
    return pollution


def get_pollution(date_start: datetime, date_end: datetime):
    pollution = pd.concat([load_pollution_file(year=year) for year in range(date_start.year, date_end.year + 1)])
    return pollution.loc[(pollution.index >= date_start) & (pollution.index <= date_end)]


if __name__ == "__main__":
    pollution = get_pollution(date_start=pd.to_datetime("2022/12/11"), date_end=pd.to_datetime("2023/03/20"))
