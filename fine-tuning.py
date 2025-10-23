from pathlib import Path
import json 
import os 
import urllib.request
import torch 
import torch.nn as nn
import numpy as np
import tensorflow as tf # noqa :(
from torch.utils.data import Dataset, DataLoader
import tiktoken
from functools import partial
import requests
from tqdm import tqdm #noqa :(
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator


device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 
print("Using device:", device ) # for me its will be GPU

#-------where to save results 

results_dir = Path("fine_tuning_results")
results_dir.mkdir(parents=True, exist_ok=True)

def download_and_load_file(file_path,url):
    if not os.path.exists(file_path):
        with urllib.request.urlopen(url) as response: 
            text_data = response.read().decode("utf-8")
        with open(file_path,'w',encoding='utf-8') as file:
            file.write(text_data)
    else:
        with open(file_path,'r',encoding='utf-8') as file:
            text_data = file.read()
    with open(file_path,'r') as file:
        data = json.load(file)
    return data

# %%
file_path = "instruction-data.json"
url = (
"https://raw.githubusercontent.com/rasbt/LLMs-from-scratch"
"/main/ch07/01_main-chapter-code/instruction-data.json"
)
data = download_and_load_file(file_path, url)
print("Number of entries:", len(data))

# %%
print("Example entry:\n", data[50])

# %%
"""
Alpaca prompt style since it is one of the most
popular ones, largely because it helped define the original approach to fine-tuning.
"""
def format_input(entry):
    instruction_text = (
    f"Below is an instruction that describes a task. "
    f"Write a response that appropriately completes the request."
    f"\n\n### Instruction:\n{entry['instruction']}"
    )
    input_text = (
    f"\n\n### Input:\n{entry['input']}" if entry["input"] else ""
    )
    return instruction_text + input_text

# %%
model_input = format_input(data[50])
desired_response = f"\n\n### Response:\n{data[50]['output']}"
print(model_input + desired_response)

# %%
model_input = format_input(data[999])
desired_response = f"\n\n### Response:\n{data[999]['output']}"
print(model_input + desired_response)

# %%
train_portion = int(len(data) * 0.85)
test_portion = int(len(data) * 0.1) 
val_portion = len(data) - train_portion - test_portion 
train_data = data[:train_portion] 
test_data = data[train_portion:train_portion + test_portion] 
val_data = data[train_portion + test_portion:] 
print("Training set length:", len(train_data))
print("Validation set length:", len(val_data))
print("Test set length:", len(test_data))

# %%
import torch 
from torch.utils.data import Dataset, DataLoader

# %%
class InstructionDataset(Dataset):
    def __init__(self,data,tokenizer): 
        self.data = data
        self.encoded_texts = []
        for entry in data:
            instruction_plus_input = format_input(entry)
            response_text = f"\n\n### Response:\n{entry['output']}"
            full_text = instruction_plus_input + response_text
            self.encoded_texts.append(
                tokenizer.encode(full_text)
            )
    def __getitem__(self,index):
        return self.encoded_texts[index]
    def __len__(self):
        return len(self.data)
        

# %%
import tiktoken
tokenizer = tiktoken.get_encoding("gpt2")
print(tokenizer.encode("<|endoftext|>", allowed_special={"<|endoftext|>"}))

# %%
def custom_collate_draft_1(
    batch,
    pad_token_id=50256,
    device="cuda" if torch.cuda.is_available() else "cpu"
):
    batch_max_lenght = max(len(item)+1 for item in batch)
    inputs_lst = [] 
    for item in batch: 
        new_item = item.copy() 
        new_item += [pad_token_id] 
        padded = (
            new_item + [ pad_token_id] * (batch_max_lenght - len(new_item))
        )
        inputs = torch.tensor(padded[:-1]) 
        inputs_lst.append(inputs)
    inputs_tensor = torch.stack(inputs_lst).to(device)
    return inputs_tensor 


# %%
inputs_1 = [0, 1, 2, 3, 4]
inputs_2 = [5, 6]
inputs_3 = [7, 8, 9]
batch = (
inputs_1,
inputs_2,
inputs_3
)
print(custom_collate_draft_1(batch))





# %%
def custom_collate_draft_2(
    batch,
    pad_token_id=50256,
    device="cuda" if torch.cuda.is_available() else "cpu"
):
    batch_max_length = max(len(item)+1 for item in batch)
    inputs_lst , targets_lst = [],[]
    for item in batch:
        new_item = item.copy() # its a tensor :)
        new_item += [pad_token_id]
        padded =(
            new_item +[ pad_token_id] * (batch_max_length - len(new_item))
            
        )
        inputs = torch.tensor(padded[:-1])
        targets = torch.tensor(padded[1:])
        inputs_lst.append(inputs)
        targets_lst.append(targets)
    inputs_tensor = torch.stack(inputs_lst).to(device)
    targets_tensor = torch.stack(targets_lst).to(device)
    return inputs_tensor,targets_tensor

# %%
inputs, targets = custom_collate_draft_2(batch)
print(inputs)
print(targets)

# %%
def custom_collate_fn(
    batch,
    pad_token_id=50256,
    ignore_index = -100,
    allowed_max_length = None,
    device="cuda" if torch.cuda.is_available() else "cpu"
):
    batch_max_length = max(len(item)+1 for item in batch)
    inputs_lst , targets_lst = [],[]
    for item in batch:
        new_item = item.copy() # its a tensor :)
        new_item += [pad_token_id]
        padded =(
            new_item +[ pad_token_id] * (batch_max_length - len(new_item))
            
        )
        inputs = torch.tensor(padded[:-1])
        targets = torch.tensor(padded[1:])
         #--------
        mask = targets == pad_token_id 
        indices = torch.nonzero(mask).squeeze() 
        if indices.numel() > 1:
            targets[indices[1:]] = ignore_index
        if allowed_max_length is not None:
            inputs = inputs[:allowed_max_length]
            targets = targets[:allowed_max_length]
        #------
        inputs_lst.append(inputs)
        targets_lst.append(targets)
    inputs_tensor = torch.stack(inputs_lst).to(device)
    targets_tensor = torch.stack(targets_lst).to(device)
    return inputs_tensor,targets_tensor

# %%
inputs, targets = custom_collate_fn(batch)
print(inputs)
print(targets)

# %%
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# %%
from functools import partial
customized_collate_fn = partial(
custom_collate_fn,
device=device,
allowed_max_length=1024
)

# %%

num_workers = 0 
batch_size = 8 # am going to be using a gpu 
torch.manual_seed(123)

train_dataset = InstructionDataset(train_data, tokenizer)
train_loader = DataLoader(
train_dataset,
batch_size=batch_size,
collate_fn=customized_collate_fn,
shuffle=True,
drop_last=True,
num_workers=num_workers
)
val_dataset = InstructionDataset(val_data, tokenizer)
val_loader = DataLoader(
val_dataset,
batch_size=batch_size,
collate_fn=customized_collate_fn,
shuffle=False,
drop_last=False,
num_workers=num_workers
)
test_dataset = InstructionDataset(test_data, tokenizer)
test_loader = DataLoader(
test_dataset,
batch_size=batch_size,
collate_fn=customized_collate_fn,
shuffle=False,
drop_last=False,
num_workers=num_workers)


# %%
print("Train loader:")
for inputs, targets in train_loader:
    print(inputs.shape, targets.shape)

# %%
# LOADING A PRE-TRAINED LLM 
# Copyright (c) Sebastian Raschka under Apache License 2.0 (see LICENSE.txt).
# Source for "Build a Large Language Model From Scratch"
#   - https://www.manning.com/books/build-a-large-language-model-from-scratch
# Code: https://github.com/rasbt/LLMs-from-scratch


import os

import requests
import json
import numpy as np
import tensorflow as tf
from tqdm import tqdm


def download_and_load_gpt2(model_size, models_dir):
    # Validate model size
    allowed_sizes = ("124M", "355M", "774M", "1558M")
    if model_size not in allowed_sizes:
        raise ValueError(f"Model size not in {allowed_sizes}")

    # Define paths
    model_dir = os.path.join(models_dir, model_size)
    base_url = "https://openaipublic.blob.core.windows.net/gpt-2/models"
    backup_base_url = "https://f001.backblazeb2.com/file/LLMs-from-scratch/gpt2"
    filenames = [
        "checkpoint", "encoder.json", "hparams.json",
        "model.ckpt.data-00000-of-00001", "model.ckpt.index",
        "model.ckpt.meta", "vocab.bpe"
    ]

    # Download files
    os.makedirs(model_dir, exist_ok=True)
    for filename in filenames:
        file_url = os.path.join(base_url, model_size, filename)
        backup_url = os.path.join(backup_base_url, model_size, filename)
        file_path = os.path.join(model_dir, filename)
        download_file(file_url, file_path, backup_url)

    # Load settings and params
    tf_ckpt_path = tf.train.latest_checkpoint(model_dir)
    settings = json.load(open(os.path.join(model_dir, "hparams.json"), "r", encoding="utf-8"))
    params = load_gpt2_params_from_tf_ckpt(tf_ckpt_path, settings)

    return settings, params


def download_file(url, destination, backup_url=None):
    def _attempt_download(download_url):
        response = requests.get(download_url, stream=True, timeout=60)
        response.raise_for_status()

        file_size = int(response.headers.get("Content-Length", 0))

        # Check if file exists and has same size
        if os.path.exists(destination):
            file_size_local = os.path.getsize(destination)
            if file_size and file_size == file_size_local:
                print(f"File already exists and is up-to-date: {destination}")
                return True

        block_size = 1024  # 1 KB
        desc = os.path.basename(download_url)
        with tqdm(total=file_size, unit="iB", unit_scale=True, desc=desc) as progress_bar:
            with open(destination, "wb") as file:
                for chunk in response.iter_content(chunk_size=block_size):
                    if chunk:
                        file.write(chunk)
                        progress_bar.update(len(chunk))
        return True

    try:
        if _attempt_download(url):
            return
    except requests.exceptions.RequestException:
        if backup_url is not None:
            print(f"Primary URL ({url}) failed. Attempting backup URL: {backup_url}")
            try:
                if _attempt_download(backup_url):
                    return
            except requests.exceptions.RequestException:
                pass

        error_message = (
            f"Failed to download from both primary URL ({url})"
            f"{' and backup URL (' + backup_url + ')' if backup_url else ''}."
            "\nCheck your internet connection or the file availability.\n"
            "For help, visit: https://github.com/rasbt/LLMs-from-scratch/discussions/273"
        )
        print(error_message)
    except Exception as e:
        print(f"An unexpected error occurred: {e}")


# Alternative way using `requests`
"""
def download_file(url, destination):
    # Send a GET request to download the file in streaming mode
    response = requests.get(url, stream=True)

    # Get the total file size from headers, defaulting to 0 if not present
    file_size = int(response.headers.get("content-length", 0))

    # Check if file exists and has the same size
    if os.path.exists(destination):
        file_size_local = os.path.getsize(destination)
        if file_size == file_size_local:
            print(f"File already exists and is up-to-date: {destination}")
            return

    # Define the block size for reading the file
    block_size = 1024  # 1 Kilobyte

    # Initialize the progress bar with total file size
    progress_bar_description = url.split("/")[-1]  # Extract filename from URL
    with tqdm(total=file_size, unit="iB", unit_scale=True, desc=progress_bar_description) as progress_bar:
        # Open the destination file in binary write mode
        with open(destination, "wb") as file:
            # Iterate over the file data in chunks
            for chunk in response.iter_content(block_size):
                progress_bar.update(len(chunk))  # Update progress bar
                file.write(chunk)  # Write the chunk to the file
"""


def load_gpt2_params_from_tf_ckpt(ckpt_path, settings):
    # Initialize parameters dictionary with empty blocks for each layer
    params = {"blocks": [{} for _ in range(settings["n_layer"])]}

    # Iterate over each variable in the checkpoint
    for name, _ in tf.train.list_variables(ckpt_path):
        # Load the variable and remove singleton dimensions
        variable_array = np.squeeze(tf.train.load_variable(ckpt_path, name))

        # Process the variable name to extract relevant parts
        variable_name_parts = name.split("/")[1:]  # Skip the 'model/' prefix

        # Identify the target dictionary for the variable
        target_dict = params
        if variable_name_parts[0].startswith("h"):
            layer_number = int(variable_name_parts[0][1:])
            target_dict = params["blocks"][layer_number]

        # Recursively access or create nested dictionaries
        for key in variable_name_parts[1:-1]:
            target_dict = target_dict.setdefault(key, {})

        # Assign the variable array to the last key
        last_key = variable_name_parts[-1]
        target_dict[last_key] = variable_array

    return params


# %%
import torch 
import torch.nn as nn 


# %%

class MultiHeadAttention(nn.Module):
    def __init__(self, d_in, d_out, context_length, dropout, num_heads, qkv_bias=False):
        super().__init__()
        assert d_out % num_heads == 0, "d_out must be divisible by num_heads"

        self.d_out = d_out
        self.num_heads = num_heads
        self.head_dim = d_out // num_heads

        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)

        self.out_proj = nn.Linear(d_out, d_out)
        self.dropout = nn.Dropout(dropout)

        mask = torch.triu(torch.ones(context_length, context_length), diagonal=1)
        self.register_buffer("mask", mask)

    def forward(self, x):
        B, T, D_in = x.shape

        Q = self.W_query(x)  # (B, T, D_out)
        K = self.W_key(x)    # (B, T, D_out)
        V = self.W_value(x)  # (B, T, D_out)


        Q = Q.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)  # (B, num_heads, T, head_dim)
        K = K.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)

        # Attention scores
        attn_scores = (Q @ K.transpose(-2, -1)) / (self.head_dim ** 0.5)  # (B, num_heads, T, T)

        mask = self.mask[:T, :T].bool()
        attn_scores = attn_scores.masked_fill(mask[None, None, :, :], float('-inf'))

        # Softmax and dropout
        attn_weights = torch.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # Apply attention
        context = attn_weights @ V  # (B, num_heads, T, head_dim)

        # Merge heads back
        context = context.transpose(1, 2).contiguous().view(B, T, self.d_out)
        context = self.out_proj(context)

        return context


# %%
class GELU(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        return 0.5 * x * (1 + torch.tanh(
        torch.sqrt(torch.tensor(2.0 / torch.pi)) *
        (x + 0.044715 * torch.pow(x, 3))
        ))

# %%
class FeedForward(nn.Module): 
    def __init__(self,cfg:dict): 
        super().__init__() 
        self.layers = nn.Sequential(
            nn.Linear(cfg['emb_dim'],4*cfg['emb_dim']),
            GELU(), 
            nn.Linear(4 * cfg['emb_dim'], cfg['emb_dim'])
        )
    
    def forward(self,x): 
        return self.layers(x) 


# %%
class LayerNorm(nn.Module): 
    def __init__(self,emb_dim): 
        super().__init__() 
        self.eps = 1e-5 
        self.scale = nn.Parameter(torch.ones(emb_dim)) 
        self.shift = nn.Parameter(torch.zeros(emb_dim)) 
    
    def forward(self,x): 
        mean = x.mean(dim=-1,keepdim=True) 
        var = x.var(dim=-1,keepdim=True,unbiased=False) 
        norm_x  = (x - mean) / torch.sqrt(var + self.eps)
        return self.scale * norm_x + self.shift # we are not forcing them to be gausian , model 
        # can do what it whant here :) 



# %%
class TransformerBlock(nn.Module): 
    def __init__(self,cfg:dict):
        super().__init__() 
        self.att = MultiHeadAttention(
            d_in= cfg['emb_dim'],
            d_out = cfg['emb_dim'], 
            context_length=cfg['context_length'], 
            num_heads = cfg['n_heads'], 
            dropout = cfg['drop_rate'], 
            qkv_bias=cfg['qkv_bias'] 
        )
        self.ff = FeedForward(cfg) 
        self.norm1 = LayerNorm(cfg['emb_dim']) 
        self.norm2 = LayerNorm(cfg['emb_dim']) 
        self.drop_shortcut = nn.Dropout(cfg['drop_rate']) 
    
    def forward(self,x): 
        shortcut = x 
        x = self.norm1(x) 
        x = self.att(x) 
        x = self.drop_shortcut(x) 
        x = x + shortcut 

        shortcut = x 
        x = self.norm2(x) 
        x = self.ff(x)  
        x = self.drop_shortcut(x) 
        x = x + shortcut 
        return x 


# %%
def assign(left, right):
    if left.shape != right.shape:
        raise ValueError(f"Shape mismatch. Left: {left.shape}, "
        "Right: {right.shape}"
    )
    return torch.nn.Parameter(torch.tensor(right))

# %%
class GPTModel(nn.Module): 
    def __init__(self,cfg:dict): 
        super().__init__() 
        self.tok_emb = nn.Embedding(cfg['vocab_size'],cfg['emb_dim']) 
        self.pos_emb = nn.Embedding(cfg['context_length'],cfg['emb_dim']) 
        self.drop_emb = nn.Dropout(cfg['drop_rate']) 

        self.trf_blocks = nn.Sequential(
            *[TransformerBlock(cfg) for _ in range(cfg['n_layers'])]
        )
        self.final_norm = LayerNorm(cfg['emb_dim']) 

        self.out_head = nn.Linear( 
            cfg['emb_dim'], cfg['vocab_size'],bias=False
        )
    
    def forward(self,in_idx:torch.Tensor): 
        batch_size, sq_len =  in_idx.shape 
        tok_embeds = self.tok_emb(in_idx)  
        pos_embeds = self.pos_emb(
            torch.arange(sq_len,device=in_idx.device)
        )
        x = tok_embeds + pos_embeds 
        x = self.drop_emb(x) 
        x = self.trf_blocks(x) 
        x = self.final_norm(x) 
        logits = self.out_head(x) 
        return logits 

# %%
def generate_text_simple(model,idx,max_new_tokens,context_size): 
    for _ in range(max_new_tokens):
        idx_cond = idx[:,-context_size:] 
        with torch.no_grad(): 
            logits = model(idx_cond) 
        logits = logits[:,-1,:]
        probas = torch.softmax(logits,dim=-1) 
        idx_next = torch.argmax(probas,dim=-1,keepdim=True) 
        idx = torch.cat((idx,idx_next),dim=1) 
    return idx 

# %%
import numpy as np
def load_weights_into_gpt(gpt, params):
    gpt.pos_emb.weight = assign(gpt.pos_emb.weight, params['wpe'])
    gpt.tok_emb.weight = assign(gpt.tok_emb.weight, params['wte'])
    for b in range(len(params["blocks"])):
        q_w, k_w, v_w = np.split(
            (params["blocks"][b]["attn"]["c_attn"])["w"], 3, axis=-1)
        gpt.trf_blocks[b].att.W_query.weight = assign(
            gpt.trf_blocks[b].att.W_query.weight, q_w.T)
        gpt.trf_blocks[b].att.W_key.weight = assign(
            gpt.trf_blocks[b].att.W_key.weight, k_w.T)
        gpt.trf_blocks[b].att.W_value.weight = assign(
            gpt.trf_blocks[b].att.W_value.weight, v_w.T)
        q_b, k_b, v_b = np.split(
            (params["blocks"][b]["attn"]["c_attn"])["b"], 3, axis=-1)
        gpt.trf_blocks[b].att.W_query.bias = assign(
            gpt.trf_blocks[b].att.W_query.bias, q_b)
        gpt.trf_blocks[b].att.W_key.bias = assign(
            gpt.trf_blocks[b].att.W_key.bias, k_b)
        gpt.trf_blocks[b].att.W_value.bias = assign(
            gpt.trf_blocks[b].att.W_value.bias, v_b)
        gpt.trf_blocks[b].att.out_proj.weight = assign(
            gpt.trf_blocks[b].att.out_proj.weight,
            params["blocks"][b]["attn"]["c_proj"]["w"].T)
        gpt.trf_blocks[b].att.out_proj.bias = assign(
            gpt.trf_blocks[b].att.out_proj.bias,
            params["blocks"][b]["attn"]["c_proj"]["b"])
        gpt.trf_blocks[b].ff.layers[0].weight = assign(
            gpt.trf_blocks[b].ff.layers[0].weight,
            params["blocks"][b]["mlp"]["c_fc"]["w"].T)
        gpt.trf_blocks[b].ff.layers[0].bias = assign(
            gpt.trf_blocks[b].ff.layers[0].bias,
            params["blocks"][b]["mlp"]["c_fc"]["b"])
        gpt.trf_blocks[b].ff.layers[2].weight = assign(
            gpt.trf_blocks[b].ff.layers[2].weight,
            params["blocks"][b]["mlp"]["c_proj"]["w"].T)
        gpt.trf_blocks[b].ff.layers[2].bias = assign(
            gpt.trf_blocks[b].ff.layers[2].bias,
            params["blocks"][b]["mlp"]["c_proj"]["b"])
        gpt.trf_blocks[b].norm1.scale = assign(
            gpt.trf_blocks[b].norm1.scale,
            params["blocks"][b]["ln_1"]["g"])
        gpt.trf_blocks[b].norm1.shift = assign(
            gpt.trf_blocks[b].norm1.shift,
            params["blocks"][b]["ln_1"]["b"])
        gpt.trf_blocks[b].norm2.scale = assign(
            gpt.trf_blocks[b].norm2.scale,
            params["blocks"][b]["ln_2"]["g"])
        gpt.trf_blocks[b].norm2.shift = assign(
            gpt.trf_blocks[b].norm2.shift,
            params["blocks"][b]["ln_2"]["b"])
    gpt.final_norm.scale = assign(gpt.final_norm.scale, params["g"])
    gpt.final_norm.shift = assign(gpt.final_norm.shift, params["b"])
    gpt.out_head.weight = assign(gpt.out_head.weight, params["wte"])

# %%
BASE_CONFIG = {
"vocab_size": 50257,
"context_length": 1024,
"drop_rate": 0.0,
"qkv_bias": True
}
# Vocabulary size
# Context length
# Dropout rate
# Query-key-value bias
model_configs = {
"gpt2-small (124M)": {"emb_dim": 768, "n_layers": 12, "n_heads": 12},
"gpt2-medium (355M)": {"emb_dim": 1024, "n_layers": 24, "n_heads": 16},
"gpt2-large (774M)": {"emb_dim": 1280, "n_layers": 36, "n_heads": 20},
"gpt2-xl (1558M)": {"emb_dim": 1600, "n_layers": 48, "n_heads": 25},
}
CHOOSE_MODEL = "gpt2-medium (355M)"
BASE_CONFIG.update(model_configs[CHOOSE_MODEL])
model_size = CHOOSE_MODEL.split(" ")[-1].lstrip("(").rstrip(")")
settings, params = download_and_load_gpt2(
model_size=model_size,
models_dir="gpt2"
)
model = GPTModel(BASE_CONFIG)
load_weights_into_gpt(model, params)
model.eval();

# %%
torch.manual_seed(123)
input_text = format_input(val_data[0])
print(input_text)

# %%
def text_to_token_ids(text,tokenizer): 
    encoded = tokenizer.encode(text,allowed_special={"<|endoftext|>"}) 
    encoded_tensor = torch.tensor(encoded).unsqueeze(0) # add batch dimension
    return encoded_tensor 

# %%
def token_ids_to_text(token_ids:torch.Tensor,tokenizer): 
    flat = token_ids.squeeze(0) # remove batch dimenstion 
    return tokenizer.decode(flat.tolist())

# %%
def generate(model, idx, max_new_tokens, context_size, temperature=0.0, top_k=None, eos_id=None):
    """
    Generate text using the model with improved memory handling and error checking.
    
    Args:
        model: The language model
        idx: Input token indices
        max_new_tokens: Maximum number of tokens to generate
        context_size: Size of the context window
        temperature: Sampling temperature (0.0 for greedy)
        top_k: Number of top tokens to consider for sampling
        eos_id: End of sequence token ID
    
    Returns:
        Generated token indices
    """
    try:
        model.eval()  # Ensure model is in evaluation mode
        
        for _ in tqdm(range(max_new_tokens), desc="Generating"):
            # Get the last context_size tokens
            idx_cond = idx[:, -context_size:]
            
            # Generate logits
            with torch.no_grad():
                logits = model(idx_cond)
                logits = logits[:, -1, :]  # Get the last token's logits
            
            # Apply top-k filtering
            if top_k is not None and top_k > 0:
                top_logits, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                min_val = top_logits[:, -1].unsqueeze(-1).expand_as(logits)
                logits = torch.where(
                    logits < min_val,
                    torch.tensor(float('-inf')).to(logits.device),
                    logits
                )
            
            # Apply temperature and sample
            if temperature > 0.0:
                logits = logits / temperature
                probs = torch.softmax(logits, dim=-1)
                idx_next = torch.multinomial(probs, num_samples=1)
            else:
                idx_next = torch.argmax(logits, dim=-1, keepdim=True)
            
            # Check for EOS token
            if eos_id is not None and (idx_next == eos_id).any():
                break
            
            # Append new token
            idx = torch.cat((idx, idx_next), dim=1)
            
            # Optional: Clear GPU cache periodically
            if torch.cuda.is_available() and _ % 100 == 0:
                torch.cuda.empty_cache()
    
    except RuntimeError as e:
        if "out of memory" in str(e):
            print("WARNING: GPU out of memory during generation")
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            # Try to recover by trimming context
            if idx.size(1) > context_size:
                idx = idx[:, -context_size:]
        else:
            raise e
    
    finally:
        model.train()  # Reset model to training mode
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    return idx
    

# %%
from base64 import decode


def generate_and_print_sample(model, tokenizer, device, start_context):
    model.eval() # evaluation mode 
    context_size = model.pos_emb.weight.shape[0] 
    encoded = text_to_token_ids(start_context,tokenizer).to(device) 
    with torch.no_grad(): 
        token_ids = generate_text_simple(
            model=model,idx=encoded,max_new_tokens=50,context_size=context_size
        )
    
    decoded_text = token_ids_to_text(token_ids,tokenizer) 
    print(decoded_text.replace("\n",""))
    model.train # back to train mode 

# %%
def evaluate_model(model,train_loader,val_loader,device,eval_iter): 
    model.eval() 
    with torch.inference_mode(): # modern 
        train_loss = calc_loss_loader(
            train_loader,model,device,num_batches=eval_iter
        )
        val_loss = calc_loss_loader(
            val_loader,model,device,num_batches=eval_iter
        )
    model.train() # turn back model to train mode 
    return train_loss, val_loss 

# %%
token_ids = generate(
model=model,
idx=text_to_token_ids(input_text, tokenizer),
max_new_tokens=35,
context_size=BASE_CONFIG["context_length"],
eos_id=50256,
)
generated_text = token_ids_to_text(token_ids, tokenizer)

# %%
response_text = generated_text[len(input_text):].strip()
print(response_text)

# %%
def calc_loss_batch(input_batch,target_batch,model,device): 
    input_batch = input_batch.to(device) 
    target_batch = target_batch.to(device) 
    logits = model(input_batch)
    loss = torch.nn.functional.cross_entropy(
        logits.flatten(0,1),target_batch.flatten()
    )
    return loss 

# %%
# Fine-tuning the LLM on instruction data
def calc_loss_loader(data_loader,model,device,num_batches=None): 
    total_loss = 0 
    if len(data_loader) == 0: 
        return float("nan") 
    elif num_batches is None: 
        num_batches = len(data_loader) 
    else: 
        num_batches = min(num_batches,len(data_loader)) 
    
    for i ,(input_batch,target_batch) in enumerate(data_loader): 
        if i < num_batches: 
            loss = calc_loss_batch(
                input_batch,target_batch,model,device
            ) 
            total_loss += loss.item() 
        else: 
            break 

    return total_loss / num_batches 

# %%
# Training an LLM 
def train_model_simple(model, train_loader, val_loader, optimizer, device, 
                      num_epochs, eval_freq, eval_iter, start_context, tokenizer):
    print("Using device:", device)
    model.to(device)
    
    # Initialize tracking variables
    train_losses, val_losses, track_tokens_seen = [], [], []
    tokens_seen, global_step = 0, -1
    best_val_loss = float('inf')
    
    try:
        for epoch in range(num_epochs):
            model.train()
            epoch_loss = 0.0
            
            # Training loop with progress bar
            with tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}') as pbar:
                for input_batch, target_batch in pbar:
                    try:
                        # Clear gradients
                        optimizer.zero_grad()
                        
                        # Forward pass
                        loss = calc_loss_batch(input_batch, target_batch, model, device)
                        
                        # Backward pass
                        loss.backward()
                        
                        # Gradient clipping to prevent exploding gradients
                        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                        
                        # Optimizer step
                        optimizer.step()
                        
                        # Update tracking
                        tokens_seen += input_batch.numel()
                        global_step += 1
                        epoch_loss += loss.item()
                        
                        # Update progress bar
                        pbar.set_postfix({'loss': f'{loss.item():.3f}'})
                        
                        # Evaluation step
                        if global_step % eval_freq == 0:
                            train_loss, val_loss = evaluate_model(
                                model, train_loader, val_loader, device, eval_iter
                            )
                            
                            train_losses.append(train_loss)
                            val_losses.append(val_loss)
                            track_tokens_seen.append(tokens_seen)
                            
                            # Save best model
                            if val_loss < best_val_loss:
                                best_val_loss = val_loss
                                torch.save({
                                    'epoch': epoch,
                                    'model_state_dict': model.state_dict(),
                                    'optimizer_state_dict': optimizer.state_dict(),
                                    'train_loss': train_loss,
                                    'val_loss': val_loss,
                                }, 'best_model_checkpoint.pth')
                            
                            print(
                                f"\nEpoch {epoch + 1} (Step {global_step:06d}): "
                                f"Train loss {train_loss:.3f}, "
                                f"Val loss {val_loss:.3f}"
                            )
                    
                    except RuntimeError as e:
                        if "out of memory" in str(e):
                            print(f'\nWARNING: out of memory, clearing cache')
                            if torch.cuda.is_available():
                                torch.cuda.empty_cache()
                            continue
                        else:
                            raise e
            
            # Generate sample at end of epoch
            generate_and_print_sample(model, tokenizer, device, start_context)
            
    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
    except Exception as e:
        print(f"\nError during training: {str(e)}")
        raise
    finally:
        # Clean up
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    return train_losses, val_losses, track_tokens_seen



# %%
model.to(device)
torch.manual_seed(123)
with torch.no_grad():
    train_loss = calc_loss_loader(
    train_loader, model, device, num_batches=5
    )
    val_loss = calc_loss_loader(
    val_loader, model, device, num_batches=5
    )
print("Training loss:", train_loss)
print("Validation loss:", val_loss)

# %%
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
def plot_losses(epochs_seen, tokens_seen, train_losses, val_losses):
    fig, ax1 = plt.subplots(figsize=(5, 3))
    ax1.plot(epochs_seen, train_losses, label="Training loss")
    ax1.plot(
    epochs_seen, val_losses, linestyle="-.", label="Validation loss"
    )
    ax1.set_xlabel("Epochs")
    ax1.set_ylabel("Loss")
    ax1.legend(loc="upper right")
    ax1.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax2 = ax1.twiny()
    ax2.plot(tokens_seen, train_losses, alpha=0)
    ax2.set_xlabel("Tokens seen")
    fig.tight_layout()
    # Save the plot to the results directory
    plot_path = results_dir / "loss_plot.png"
    plt.savefig(plot_path, dpi=300)
    plt.close()
import time
start_time = time.time()
torch.manual_seed(123)
optimizer = torch.optim.AdamW(
model.parameters(), lr=0.00005, weight_decay=0.1
)
num_epochs = 2 # will guest train for two epochs 
train_losses, val_losses, tokens_seen = train_model_simple(
model, train_loader, val_loader, optimizer, device,
num_epochs=num_epochs, eval_freq=5, eval_iter=5,
start_context=format_input(val_data[0]), tokenizer=tokenizer
)
end_time = time.time()
execution_time_minutes = (end_time - start_time) / 60
print(f"Training completed in {execution_time_minutes:.2f} minutes.")



epochs_tensor = torch.linspace(0, num_epochs, len(train_losses))
plot_losses(epochs_tensor, tokens_seen, train_losses, val_losses)

torch.manual_seed(123)

# %%

# %% [markdown]
# we can evaluate our model using some of this approaches : 
# - Short-answer and multiple-choice benchmarks, such as Measuring Massive Mul-
# titask Language Understanding (MMLU; https://arxiv.org/abs/2009.03300),
# which test the general knowledge of a model.
# - Human preference comparison to other LLMs, such as LMSYS chatbot arena
# (https://arena.lmsys.org).
# 
# - Automated conversational benchmarks, where another LLM like GPT-4 is
# used to evaluate the responses, such as AlpacaEval (https://tatsu-lab.github.io/
# alpaca_eval/).
# 
# 
# in practice, it can be useful to consider all three types of evaluation methods: multiple-
# choice question answering, human evaluation, and automated metrics that measure
# conversational performance. However, since we are primarily interested in assessing con-
# versational performance rather than just the ability to answer multiple-choice ques-
# tions, human evaluation and automated metrics may be more relevant

# %%
# Generate responses for test data
def generate_and_evaluate(model, test_data, tokenizer, device, config):
    model.eval()
    results = []
    
    for i, entry in tqdm(enumerate(test_data), total=len(test_data)):
        input_text = format_input(entry)
        token_ids = generate(
            model=model,
            idx=text_to_token_ids(input_text, tokenizer).to(device),
            max_new_tokens=256,
            context_size=config["context_length"],
            eos_id=50256
        )
        generated_text = token_ids_to_text(token_ids, tokenizer)
        response_text = (
            generated_text[len(input_text):]
            .replace("### Response:", "")
            .strip()
        )
        test_data[i]["model_response"] = response_text
        results.append({"input": input_text, "response": response_text})
    
    return test_data, results

# Save model and responses
def save_model_and_responses(model, test_data, model_name):
    output_file = results_dir / "instruction-data-with-response.json"
    with open(output_file, "w") as file:
        json.dump(test_data, file, indent=4)
    
    # Save model
    import re
    file_name = f"{re.sub(r'[ ()]', '', model_name)}-sft.pth"
    file_path = results_dir / file_name
    torch.save(model.state_dict(), file_path)
    print(f"Model and responses saved: {file_path}, {output_file}")

# Run evaluation
test_data, results = generate_and_evaluate(model, test_data, tokenizer, device, BASE_CONFIG)
save_model_and_responses(model, test_data, CHOOSE_MODEL)





