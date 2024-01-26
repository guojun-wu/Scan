import argparse, json
import random
import string
import torch
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from transformers import (
    GPT2Tokenizer, 
    GPT2ForSequenceClassification ,
    BertTokenizer,
    BertForSequenceClassification,
    RobertaForSequenceClassification,
    AutoTokenizer,
    DistilBertForSequenceClassification,
    OPTForSequenceClassification,
)

plt.rcParams['figure.figsize'] = [10, 10]

# Adapted from AllenNLP Interpret and Han et al. 2020
def register_embedding_list_hook(model, embeddings_list):
    def forward_hook(module, inputs, output):
        embeddings_list.append(output.squeeze(0).clone().cpu().detach().numpy())
    # Define a dictionary mapping model classes to their embedding layer attributes
    if isinstance(model, GPT2ForSequenceClassification):
        embedding_layer = model.transformer.wte
    elif isinstance(model, BertForSequenceClassification):
        embedding_layer = model.bert.embeddings.word_embeddings
    elif isinstance(model, RobertaForSequenceClassification):
        embedding_layer = model.roberta.embeddings.word_embeddings
    elif isinstance(model, DistilBertForSequenceClassification):
        embedding_layer = model.distilbert.embeddings.word_embeddings
    elif isinstance(model, OPTForSequenceClassification):
        embedding_layer = model.get_input_embeddings()

    handle = embedding_layer.register_forward_hook(forward_hook)
    return handle

def register_embedding_gradient_hooks(model, embeddings_gradients):
    def hook_layers(module, grad_in, grad_out):
        embeddings_gradients.append(grad_out[0].detach().cpu().numpy())
    
    if isinstance(model, GPT2ForSequenceClassification):
        embedding_layer = model.transformer.wte
    elif isinstance(model, BertForSequenceClassification):
        embedding_layer = model.get_input_embeddings()
    elif isinstance(model, RobertaForSequenceClassification):
        embedding_layer = model.roberta.embeddings.word_embeddings
    elif isinstance(model, DistilBertForSequenceClassification):
        embedding_layer = model.distilbert.embeddings.word_embeddings
    elif isinstance(model, OPTForSequenceClassification):
        embedding_layer = model.get_input_embeddings()
    hook = embedding_layer.register_full_backward_hook(hook_layers)
    return hook

def merge_gpt_tokens(tokens, gradients):
    merged_gradients = []
    word = ""
    word_gradients = 0
    # Merge tokens into the original words
    for i, token in enumerate(tokens):
        if token.startswith("Ġ") or token.startswith("Ċ"):
            if word != "":
                merged_gradients.append(word_gradients)
                word = ""
                word_gradients = 0
            word = token[1:]
            word_gradients = gradients[i]
        else:
            word += token
            word_gradients += gradients[i]
    if word != "":
        merged_gradients.append(word_gradients)
    return np.array(merged_gradients).squeeze()

def merge_further(word_list, text, gradients):
    original_wrods = text.lower().split()
    merged_word_list = []
    merged_gradients = []
    tmp = ""
    word_gradients = 0
    word_count = 0
    
    for target in original_wrods:     
        for word in word_list[word_count:]:
            if word == target:
                merged_word_list.append(word)
                merged_gradients.append(gradients[word_count])
                word_count += 1
                break
            elif word in target:
                tmp += word
                word_gradients += gradients[word_count]
                word_count += 1
            else:
                merged_word_list.append(tmp)
                merged_gradients.append(word_gradients)
                tmp = ""
                word_gradients = 0
                break
    if tmp != "":
        merged_word_list.append(tmp)
        merged_gradients.append(word_gradients)
                
    return merged_gradients, merged_word_list

def merge_bert_tokens(tokens, text, gradients, special_tokens=["[CLS]", "[SEP]", "[MASK]"]):
    gradients_list = []
    word_list = []
    word = ""
    word_gradients = 0
    # Merge tokens into the original words
    for i, token in enumerate(tokens):
        if token in special_tokens:
            continue
        if token.startswith("##"):
            word += token[2:]
            word_gradients += gradients[i]
        else:
            if word != "":
                word_list.append(word)
                gradients_list.append(word_gradients)
                word = ""
                word_gradients = 0
            word = token
            word_gradients = gradients[i]
    if word != "":
        word_list.append(word)
        gradients_list.append(word_gradients)
    # Merge further the words into the original text

    merged_gradients, merged_word_list = merge_further(word_list, text, gradients_list)

    return np.array(merged_gradients).squeeze()

def lm_saliency(model, tokenizer, input_ids, input_mask, label_id):

    torch.enable_grad()
    model.eval()

    embeddings_list = []
    gradients_list = []

    # Convert input_ids and attention_mask to PyTorch tensors
    input_ids = torch.tensor(input_ids, dtype=torch.long).to(model.device)
    input_mask = torch.tensor(input_mask, dtype=torch.long).to(model.device)

    handle = register_embedding_list_hook(model, embeddings_list)
    hook = register_embedding_gradient_hooks(model, gradients_list)

    model.zero_grad()
    A = model(input_ids.unsqueeze(0), attention_mask=input_mask.unsqueeze(0))
    A.logits[0][label_id].backward()

    handle.remove()
    hook.remove()

    gradients_list = np.array(gradients_list).squeeze()
    embeddings_list = np.array(embeddings_list).squeeze()
    # Merge tokens into the original words
    tokens = tokenizer.convert_ids_to_tokens(input_ids)

    return tokens, gradients_list, embeddings_list

def input_x_gradient(tokens, input_text, grads, embds, model, normalize=False):
    input_grad = np.sum(grads * embds, axis=-1).squeeze()

    if isinstance(model, BertForSequenceClassification) or isinstance(model, DistilBertForSequenceClassification):
        input_grad = merge_bert_tokens(tokens, input_text, input_grad)
    else:
        input_grad = merge_gpt_tokens(tokens, input_grad)
    if normalize:
        norm = np.linalg.norm(input_grad, ord=1)
        input_grad /= norm
        
    return input_grad
  
    return input_grad

def l1_grad_norm(tokens, input_text, grads, model, normalize=False):
    l1_grad = np.linalg.norm(grads, ord=1, axis=-1).squeeze()
        
    if isinstance(model, BertForSequenceClassification) or isinstance(model, DistilBertForSequenceClassification):
        l1_grad = merge_bert_tokens(tokens, input_text, l1_grad)
    else:
        l1_grad = merge_gpt_tokens(tokens, l1_grad)
    
    if normalize:
        norm = np.linalg.norm(l1_grad, ord=1)
        l1_grad /= norm
    
    return l1_grad

def l2_grad_norm(tokens, input_text, grads, model, normalize=False):
    l2_grad = np.linalg.norm(grads, ord=2, axis=-1).squeeze()

    if isinstance(model, BertForSequenceClassification) or isinstance(model, DistilBertForSequenceClassification):
        l2_grad = merge_bert_tokens(tokens, input_text, l2_grad)
    else:
        l2_grad = merge_gpt_tokens(tokens, l2_grad)
    
    if normalize:
        norm = np.linalg.norm(l2_grad, ord=1)
        l2_grad /= norm
    return l2_grad
def visualize(attention, tokenizer, input_ids, gold=None, normalize=False, print_text=True, save_file=None, title=None, figsize=60, fontsize=36):
    tokens = [tokenizer.decode(i) for i in input_ids[0][:len(attention) + 1]]
    if gold is not None:
        for i, g in enumerate(gold):
            if g == 1:
                tokens[i] = "**" + tokens[i] + "**"

    # Normalize to [-1, 1]
    if normalize:
        a,b = min(attention), max(attention)
        x = 2/(b-a)
        y = 1-b*x
        attention = [g*x + y for g in attention]
    attention = np.array([list(map(float, attention))])

    fig, ax = plt.subplots(figsize=(figsize,figsize))
    norm = mpl.colors.Normalize(vmin=-1, vmax=1)
    im = ax.imshow(attention, cmap='seismic', norm=norm)

    if print_text:
        ax.set_xticks(np.arange(len(tokens)))
        ax.set_xticklabels(tokens, fontsize=fontsize)
    else:
        ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)


    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
         rotation_mode="anchor")
    for (i, j), z in np.ndenumerate(attention):
        ax.text(j, i, '{:0.2f}'.format(z), ha='center', va='center', fontsize=fontsize)


    ax.set_title("")
    fig.tight_layout()
    if title is not None:
        plt.title(title, fontsize=36)
    
    if save_file is not None:
        plt.savefig(save_file, bbox_inches = 'tight',
        pad_inches = 0)
        plt.close()
    else:
        plt.show()

def main():
    pass

if __name__ == "__main__":
    main()
