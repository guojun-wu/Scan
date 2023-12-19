import argparse, json
import random
import torch
import numpy as np
from transformers import (
    WEIGHTS_NAME,
    GPT2Config,
    GPT2Tokenizer,
    GPT2LMHeadModel,
)

import matplotlib as mpl
import matplotlib.pyplot as plt

plt.rcParams['figure.figsize'] = [10, 10]

config = GPT2Config.from_pretrained("gpt2")
VOCAB_SIZE = config.vocab_size
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

# Adapted from AllenNLP Interpret and Han et al. 2020
def register_embedding_list_hook(model, embeddings_list):
    def forward_hook(module, inputs, output):
        embeddings_list.append(output.squeeze(0).clone().cpu().detach().numpy())
    embedding_layer = model.transformer.wte
    handle = embedding_layer.register_forward_hook(forward_hook)
    return handle

def register_embedding_gradient_hooks(model, embeddings_gradients):
    def hook_layers(module, grad_in, grad_out):
        embeddings_gradients.append(grad_out[0].detach().cpu().numpy())
    embedding_layer = model.transformer.wte
    hook = embedding_layer.register_full_backward_hook(hook_layers)
    return hook

def merge_tokens(tokens, embeddings_gradients, embeddings_list):
    merged_embeddings = []
    merged_gradients = []
    word = ""
    word_gradients = 0
    word_embeddings = 0
    # Merge tokens into the original words
    for i, token in enumerate(tokens):
        if token.startswith("Ġ") or token.startswith("Ċ"):
            if word != "":
                merged_gradients.append(word_gradients)
                merged_embeddings.append(word_embeddings)
                word = ""
                word_gradients = 0
                word_embeddings = 0
            word = token[1:]
            word_gradients = embeddings_gradients[i]
            word_embeddings = embeddings_list[i]
        else:
            word += token
            word_gradients += embeddings_gradients[i]
            word_embeddings += embeddings_list[i]  
    if word != "":
        merged_gradients.append(word_gradients)
        merged_embeddings.append(word_embeddings)
    return np.array(merged_gradients).squeeze(), np.array(merged_embeddings).squeeze()
           
def lm_saliency(model, input_ids, input_mask, output_ids):
    
    torch.enable_grad()
    model.eval()

    embeddings_list = []
    gradients_list = []

    # Convert input_ids and attention_mask to PyTorch tensors
    input_ids = torch.tensor(input_ids, dtype=torch.long).to(model.device)
    input_mask = torch.tensor(input_mask, dtype=torch.long).to(model.device)
    output_ids = torch.tensor(output_ids, dtype=torch.long).to(model.device)

    # Register hooks
    handle = register_embedding_list_hook(model, embeddings_list)
    hook = register_embedding_gradient_hooks(model, gradients_list)

    # Forward pass
    A = model(input_ids, attention_mask=input_mask)
    loss = torch.nn.CrossEntropyLoss()(A.logits[-len(output_ids):].view(-1, A.logits.size(-1)), output_ids)

    # Backward pass
    model.zero_grad()
    loss.backward()

    # Remove hooks
    handle.remove()
    hook.remove()

    gradients_list = np.array(gradients_list).squeeze()
    embeddings_list = np.array(embeddings_list).squeeze()
    # Merge tokens into the original words
    tokens = tokenizer.convert_ids_to_tokens(input_ids)
    gradients_list, embeddings_list = merge_tokens(tokens, gradients_list, embeddings_list)

    return gradients_list, embeddings_list

def input_x_gradient(grads, embds, normalize=False):
    input_grad = np.sum(grads * embds, axis=-1).squeeze()

    if normalize:
        # norm = np.linalg.norm(input_grad, ord=1)
        # input_grad /= norm
        # softmax
        input_grad = np.exp(input_grad) / np.sum(np.exp(input_grad))
        
    return input_grad

def l1_grad_norm(grads, normalize=False):
    l1_grad = np.linalg.norm(grads, ord=1, axis=-1).squeeze()

    if normalize:
        norm = np.linalg.norm(l1_grad, ord=1)
        l1_grad /= norm

    return l1_grad


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
