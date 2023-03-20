from pathlib import Path

project_root = Path(__file__).parent.parent
data_dir = Path.joinpath(project_root, 'data')
results_dir = Path.joinpath(project_root, 'results')
observations_dir = Path.joinpath(data_dir, 'observation')
traffic_dir = Path.joinpath(data_dir, 'traffic')
city_dir = Path.joinpath(data_dir, "city")

data_dir.mkdir(parents=True, exist_ok=True)
results_dir.mkdir(parents=True, exist_ok=True)
observations_dir.mkdir(parents=True, exist_ok=True)
traffic_dir.mkdir(parents=True, exist_ok=True)
city_dir.mkdir(parents=True, exist_ok=True)