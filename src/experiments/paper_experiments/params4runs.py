import psutil

import src.config as config
from PerplexityLab.LaTexReports import RunsInfo2Latex
from src.lib.DataProcessing.SeleniumScreenshots import window_size, zoom
from src.lib.DataProcessing.TrafficProcessing import image_shape

runsinfo = RunsInfo2Latex(path2latex=f"{config.paper_dir}/main.tex")
runsinfo.insert_preamble_in_latex_file()
path2latex_figures = runsinfo.latex_folder.joinpath("figures")

# ----- parameters of experiment ----- #
recalculate_traffic_by_pixel = False
proportion_of_past_times = 0.8
screenshot_period = 15
# shuffle times
shuffle = False
simulation = False
filter_graph = True  # filter by nodes with neighboaring edges having traffic and keep the biggest commponent.
seed = 42
all_stations = ['OPERA', 'BP_EST', 'AUT', 'BASCH', 'BONAP', 'CELES', 'ELYS', 'PA07', 'PA12', 'PA13', 'PA18', 'HAUS',
                'PA15L']
# stations that are inside Paris so traffic information is all around
# stations2test = ['OPERA', 'HAUS', 'BONAP', 'CELES',  'ELYS', 'PA07', 'PA13', 'PA18']
stations2test = ['HAUS', 'CELES',  'PA13', 'OPERA', 'BASCH', 'PA12',  'BONAP', 'ELYS', 'PA07', 'PA18', ]
# stations2test = ['PA13', 'OPERA', 'BONAP', 'ELYS', 'PA07', 'PA18', 'BASCH', 'PA12', 'CELES', 'HAUS', 'PA15L']
# stations2test = all_stations

runsinfo.append_info(
    filtergraph=filter_graph,
    shuffletimes=shuffle,
    screenshotperiod=screenshot_period,
    screenshotwidth=window_size['width'],
    screenshotheight=window_size['height'],
    zoom=zoom,
    imagex=image_shape[0],
    imagey=image_shape[1],
    numstationstest=len(stations2test),
    stationstest=", ".join(stations2test),
    percentagetraintime=int(100 * proportion_of_past_times),
)

RAM = psutil.virtual_memory().total / 1000000000
server = RAM > 50
if server:  # if run in server
    print("running in server")
    nrows2load_traffic_data = None  # None 1000
    num_cores = 8
    chunksize = 500
else:
    print("running in local machine")
    nrows2load_traffic_data = 1000  # None 1000
    num_cores = 14
    chunksize = None
