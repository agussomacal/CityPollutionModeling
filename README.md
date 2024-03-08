This repository has the python implementation of the paper:

**State estimation of urban air pollution with statistical, physical, and super-learning graph models**: [arXiv-preprint](https://arxiv.org/abs/2402.02812) | [Poster summary](https://github.com/agussomacal/CityPollutionModeling/blob/main/poster/PollutionPoster.pdf)
<br><sub>
We consider the problem of real-time reconstruction of urban air pollution 
maps. The task is challenging due to the heterogeneous sources of available 
data, the scarcity of direct measurements, the presence of noise, and the 
large surfaces that need to be considered. In this work, we introduce 
different reconstruction methods based on posing the problem on city graphs. 
Our strategies can be classified as fully data-driven, physics-driven, or 
hybrid, and we combine them with super-learning models. The performance of 
the methods is tested in the case of the inner city of Paris, France.
</sub> <br><br>

## Running experiments

The script that produces the different figures of the paper can be found in

```
/src/experiments/paper_clean_experiments/NumericalExperiments.py
```

## Setup for developers
We recommend first to work in a virtual environment which can be created using 
previously installed python packages venv or virtualenv through
```
python3.8 -m venv venv
```
or
```
virtualenv -p python3.8 test
```

Then activate virtual environment
```
. .venv/bin/activate
```
Install required packages usually through:
```
pip install -r requirements.txt 
```
However, if this doesn't work for you, try to install them one by one in the order specified by the requirements.txt file.



