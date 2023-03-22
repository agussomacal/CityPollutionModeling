import pandas as pd


def get_traffic_pollution_data_per_hour(traffic_data, pollution, average=True):
    if average:
        normalized_stats_per_hour = traffic_data.resample('H').sum()
        normalized_stats_per_hour.index = normalized_stats_per_hour.index + pd.tseries.frequencies.to_offset("30Min")
        return normalized_stats_per_hour.loc[normalized_stats_per_hour.index.intersection(pollution.index), :], \
            pollution.loc[pollution.index.intersection(normalized_stats_per_hour.index)]
    else:
        return traffic_data.loc[traffic_data.index.intersection(pollution.index), :], \
            pollution.loc[pollution.index.intersection(traffic_data.index)]