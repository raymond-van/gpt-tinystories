import json
import numpy as np
import random
import torch 

def load_config(path):
    with open(path, 'r') as f:
        cfg = json.load(f)
    return cfg

def load_data(load_train=False, to_torch=True):
    if load_train:
        if to_torch:
            with open('train.bin', 'rb') as f:
                binary_data = f.read()
                m = np.frombuffer(binary_data, dtype=np.int32)
                m = torch.from_numpy(m)
        else:    
            m = np.memmap('train.bin', dtype='int16', mode='r')
        return m
    else:
        if to_torch:
            with open('valid.bin', 'rb') as f:
                binary_data = f.read()
                m = np.frombuffer(binary_data, dtype=np.int32)
                print(type(m))
                m = torch.from_numpy(m)
        else:   
            m = np.memmap('valid.bin', dtype='int16', mode='r')
        return m

# Only loads state dict
def load_model(cfg, path, parallel=False, device='cuda'):
    model = GPT(cfg)
    if parallel:
        model = nn.DataParallel(model)
    model.to(device)
    model.load_state_dict(path)
    return model

def test_language_modeling(model, tokenizer, len=200, prompt=None, device='cuda'):
    if prompt is None:
        prompt = "One day, a little girl named Lily found a needle in her room."
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
    if torch.cuda.device_count() > 1:
        greedy_output = model.module.generate(input_ids, len)
    else:
        greedy_output = model.generate(input_ids, len)
    print("Output:\n" + 100 * '-')
    print(tokenizer.decode(greedy_output[0], skip_special_tokens=True))

def estimate_loss(model, tokenizer, valid_loader, device='cuda'):
    model.eval()
    with torch.no_grad():
        losses = torch.zeros(40)
        for k,batch in enumerate(valid_loader):
            tokenized = tokenizer(batch['text'], padding=True, return_tensors='pt', max_length = 256, truncation = True)['input_ids'].to(device)
            _, loss = model(tokenized,tokenized)
            if torch.cuda.device_count() > 1:
                loss = loss.mean()
            losses[k] = loss.item()
            if k == 40 - 1 :
                break
    model.train()
    return losses.mean()

def save_checkpoint(model, optimizer, updates, filename="checkpoint.pt.tar"):
    state = {'updates': updates,
             'state_dict': model.state_dict(),
             'optimizer': optimizer.state_dict()}
    torch.save(state, filename)

def load_checkpoint(model, filename, optim=None):
    checkpoint = torch.load(filename)
    model.load_state_dict(checkpoint['state_dict'])
    if optim is not None:
        optim.load_state_dict(checkpoint['optimizer'])
    updates = checkpoint['updates']
    return updates

def encode_large_text_file(file_path, tokenizer, chunk_size=4096):
    # Open the file and initialize an empty list to store encoded chunks
    with open(file_path, 'r', encoding='utf-8') as file:
        encoded_chunks = []
        # Read the file in chunks
        while True:
            chunk = file.read(chunk_size)
            # Break the loop if the end of the file is reached
            if not chunk:
                break
            # Encode the chunk using the tokenizer
            encoded_chunk = tokenizer.encode(chunk, return_tensors="pt")
            encoded_chunks.append(encoded_chunk)
    # Concatenate the list of encoded chunks along the sequence dimension
    final_encoded_output = torch.cat(encoded_chunks, dim=-1)
    return final_encoded_output.numpy().astype(np.uint16)

def setup_seed(seed):
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     np.random.seed(seed)
     random.seed(seed)
     torch.backends.cudnn.deterministic = True
