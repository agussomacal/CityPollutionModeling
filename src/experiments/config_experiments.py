import psutil

# ----- parameters of experiment ----- #
recalculate_traffic_by_pixel = False
proportion_of_past_times = 0.8
screenshot_period = 15
shuffle = False
simulation = False
max_num_stations = 100
filter_graph = True  # filter by nodes with neighboaring edges having traffic and keep the biggest commponent.
seed = 42
# stations2test = ['OPERA', 'BP_EST', 'AUT', 'BASCH', 'BONAP', 'CELES', 'ELYS', 'PA07', 'PA12', 'PA13', 'PA18', 'HAUS',
#                  'PA15L']
# stations that are inside Paris so traffic information is all around
# stations2test = ['OPERA', 'HAUS', 'BONAP', 'CELES',  'ELYS', 'PA07', 'PA13', 'PA18']
stations2test = ['OPERA', 'BONAP', 'ELYS', 'PA07', 'PA13', 'PA18', 'BASCH', 'PA12', ]

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

if simulation:
    import datetime

    from src.lib.Models.BaseModel import ModelsAggregator
    from src.lib.Models.TrueStateEstimationModels.AverageModels import GlobalMeanModel
    from src.lib.Models.TrueStateEstimationModels.TemporalDependentModels import CosinusModel
    from src.lib.Models.TrueStateEstimationModels.TrafficConvolution import TrafficConvolutionModel, \
        gaussker

    simulated_model = ModelsAggregator(
        models=[
            GlobalMeanModel(global_mean=60),
            CosinusModel(t0=datetime.datetime(year=2023, month=1, day=1, hour=8), amplitude=40, period=12, phase=0),
            TrafficConvolutionModel(conv_kernel=gaussker, normalize=True,
                                    sigma=0.05, green=0.4, yellow=5, red=26, dark_red=64.46007627922917)
        ],
        weights=[1, 1, 1, 0]
    )
