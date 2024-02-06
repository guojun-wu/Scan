## Install required packages

To install the necessary Python packages, navigate to the Scan directory and execute the following command:

```bash
cd Scan
pip install -r requirements.txt
```

## Download eye-tracking data and prepare the data

Data will be downloaded from the [OSF repository](https://osf.io/q3zws/) and saved in the `data/zuco` directory. Run the following script to download and prepare the data:

```bash
pre/get_zuco_data.sh
```

## Fine-tune the model

To fine-tune the model, run the finetune.py script with the appropriate task and model names. Replace task_name and model_name with your specific task and model:

```bash
python finetune.py  -t task_name -m model_name
```

## Get gradient-based saliency

The gradient-based saliency can be obtained using the lm.py script. Use the --tuned flag to indicate whether the model is fine-tuned, pre-trained, or randomly initialized. Replace task_name and model_name with your specific task and model. For example:

```bash
python lm.py -t task_name -m model_name --tuned finetuned
```

## The analysis of correlation between saliency and human fixation

For detailed analysis of the correlation between saliency and human fixation, refer to the [view.ipynb](view.ipynb) notebook.
