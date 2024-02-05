import requests

def get_config(model_name):
    url_dict = {
        "gpt2": "https://huggingface.co/gpt2/resolve/main/config.json",
        "gpt2_large": "https://huggingface.co/gpt2-large/resolve/main/config.json",
        "distilbert": "https://huggingface.co/distilbert-base-uncased/resolve/main/config.json",
        "bert": "https://huggingface.co/bert-base-uncased/resolve/main/config.json",
        "bert_large": "https://huggingface.co/bert-large-uncased/resolve/main/config.json",
        "roberta": "https://huggingface.co/roberta-base/resolve/main/config.json",
        "opt": "https://huggingface.co/facebook/opt-350m/resolve/main/config.json",
    }

    url = url_dict[model_name]
    response = requests.get(url)
    if response.status_code == 200:
        json_data = response.json()
    else:
        
        print("Failed to retrieve JSON data. Status code:", response.status_code)
    return url

subj_sst_acc = {"ZAB": 76.09,"ZDM": 76.09,"ZDN": 89.13, "ZGW": 71.74, "ZJM": 80.43, 
                "ZJN": 54.34, "ZJS":91.30, "ZKB":89.13, "ZKH":76.09, "ZKW":69.57, "ZMG":91.30, "ZPH":89.13}
subj_wiki_acc = {"ZAB": 90.42, "ZDM": 96.81, "ZDN": 92.87, "ZGW": 92.14, "ZJM": 79.12,
                "ZJN": 96.56, "ZJS": 93.86, "ZKB": 95.33, "ZKH": 93.12, "ZKW": 94.84, "ZMG": 95.82, "ZPH": 97.05}
task_title = {"sst": "Sentiment Analysis", "wiki": "Relation Extraction"}

