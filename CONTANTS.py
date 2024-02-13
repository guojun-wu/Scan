
DATA_PATH = "data" 
CHECKPOINT_PATH = "checkpoints" 
RESULT_PATH = "results" 

url_dict = {
        "gpt2": "https://huggingface.co/gpt2/resolve/main/config.json",
        "gpt2_large": "https://huggingface.co/gpt2-large/resolve/main/config.json",
        "distilbert": "https://huggingface.co/distilbert-base-uncased/resolve/main/config.json",
        "bert": "https://huggingface.co/bert-base-uncased/resolve/main/config.json",
        "bert_large": "https://huggingface.co/bert-large-uncased/resolve/main/config.json",
        "roberta": "https://huggingface.co/roberta-base/resolve/main/config.json",
        "opt": "https://huggingface.co/facebook/opt-350m/resolve/main/config.json",
    }
subj_sst_acc = {"ZAB": 76.09,"ZDM": 76.09,"ZDN": 89.13, "ZGW": 71.74, "ZJM": 80.43, 
                "ZJN": 54.34, "ZJS":91.30, "ZKB":89.13, "ZKH":76.09, "ZKW":69.57, "ZMG":91.30, "ZPH":89.13}
subj_wiki_acc = {"ZAB": 90.42, "ZDM": 96.81, "ZDN": 92.87, "ZGW": 92.14, "ZJM": 79.12,
                "ZJN": 96.56, "ZJS": 93.86, "ZKB": 95.33, "ZKH": 93.12, "ZKW": 94.84, "ZMG": 95.82, "ZPH": 97.05}
task_title = {"sst": "Sentiment Analysis", "wiki": "Relation Extraction"}

path_dict = {
        "bert": "bert-base-uncased", 
        "bert_large": "bert-large-uncased",
        "roberta": "roberta-base", 
        "gpt2": "gpt2", 
        "gpt2_large": "gpt2-large",
        "distilbert": "distilbert-base-uncased",
        "opt": "facebook/opt-350m"}

num_dict = {"sst": 3, "wiki": 9}

