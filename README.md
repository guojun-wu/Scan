# This is the repository for the seminar project on Cognitively Enhanced NLP

# The repository is structured as follows

# - `data/`: contains the data used for the project

# - `pre/`: contains the scripts used to download and preprocess the data

## Setup

Install the requirements using `pip install -r requirements.txt`

## Data

download the data using `bash pre/get_zuco_data.sh`

## Preprocessing

Preprocess the data using `python pre/preprocess.py --zuco_task zuco11`

## Get input saliency

Get the input saliency using `python lm.py -t zuco11 -m bert`

## Compute the spearman correlation

Compute the spearman correlation using `python corr.py -t zuco11 -m bert`
