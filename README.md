# Setup

## Install required packages

    ```bash
    cd Scan
    pip install -r requirements.txt
    ```

## Download eye-tracking data and prepare the data

Data will be downloaded from the OSF repository[https://osf.io/q3zws/] and saved in the `data/zuco` directory.

    ```bash
    pre/get_zuco_data.sh
    ```

## Fine-tune the model

    ```bash
    python finetune.py  -t task_name -m model_name
    ```

## Get gradient-based saliency

    ```bash
    python lm.py -t task_name -m model_name --tuned 
    ```
