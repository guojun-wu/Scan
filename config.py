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


