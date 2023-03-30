import logging

import psutil
import seaborn as sns

import src.config as config

sns.set_theme()

# ----- parameters of experiment ----- #
recalculate_traffic_by_pixel = False
proportion_of_past_times = 0.8
screenshot_period = 15

RAM = psutil.virtual_memory().total / 1000000000
if RAM > 100:  # if run in server
    nrows2load_traffic_data = None  # None 1000
    num_cores = 1
else:
    nrows2load_traffic_data = 500  # None 1000
    num_cores = 10

# ----- logger ----- #
# Create and configure logger
logging.basicConfig(
    level=logging.INFO,
    # handlers=[logging.FileHandler(f"{data_manager.path}/experiment.log"), logging.StreamHandler()],
    filename=f"{config.results_dir}/experiment.log",
    format='%(asctime)s %(message)s',
    filemode='a')

# define a Handler which writes INFO messages or higher to the sys.stderr
console = logging.StreamHandler()
console.setLevel(logging.INFO)
# set a format which is simpler for console use
formatter = logging.Formatter('%(name)-12s: %(levelname)-8s %(message)s')
# tell the handler to use this format
console.setFormatter(formatter)
# add the handler to the root logger
logging.getLogger().addHandler(console)
