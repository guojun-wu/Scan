import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel

tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
model = GPT2LMHeadModel.from_pretrained("gpt2")
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
model.to(device)

input = "Can you stop the dog from "
input_tokens = tokenizer(input)['input_ids']
attention_ids = tokenizer(input)['attention_mask']

from lm_saliency import *

target = "barking"
foil = "crying"
CORRECT_ID = tokenizer(" "+ target)['input_ids'][0]
FOIL_ID = tokenizer(" "+ foil)['input_ids'][0]

base_saliency_matrix, base_embd_matrix = saliency(model, input_tokens, attention_ids)
saliency_matrix, embd_matrix = saliency(model, input_tokens, attention_ids, foil=FOIL_ID)

# Input x gradient
base_explanation = input_x_gradient(base_saliency_matrix, base_embd_matrix, normalize=True)
contra_explanation = input_x_gradient(saliency_matrix, embd_matrix, normalize=True)

# Gradient norm
base_explanation = l1_grad_norm(base_saliency_matrix, normalize=True)
contra_explanation = l1_grad_norm(saliency_matrix, normalize=True)

# Erasure
base_explanation = erasure_scores(model, input_tokens, attention_ids, normalize=True)
contra_explanation = erasure_scores(model, input_tokens, attention_ids, correct=CORRECT_ID, foil=FOIL_ID, normalize=True)

visualize(np.array(base_explanation), tokenizer, [input_tokens], print_text=True, title=f"Why did the model predict {target}?")
visualize(np.array(contra_explanation), tokenizer, [input_tokens], print_text=True, title=f"Why did the model predict {target} instead of {foil}?")


