from saliency import *
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
model = GPT2LMHeadModel.from_pretrained("gpt2")
input_seq = "Can you stop the dog from barking out-bad-act" 
output_seq = "loudly"

# tokenize output sequences
output_tokens = tokenizer(output_seq)['input_ids']

# add whitespace to input_seq for tokens in output_seq
input_seq = input_seq.strip() + " " * (len(output_tokens))

# tokenize input sequence
input_tokens = tokenizer(input_seq)['input_ids']
attention_ids = tokenizer(input_seq)['attention_mask']

base_saliency_matrix, base_embd_matrix = seq_saliency(model, input_tokens, attention_ids, output_tokens)
base_explanation = input_x_gradient(base_saliency_matrix, base_embd_matrix, normalize=True)
print(base_explanation)
