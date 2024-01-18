import torch
import torch.nn as nn
from torch.nn import functional as F

class GPT(nn.Module):
    # GPT2 implementation
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.wte = nn.Embedding(config['vocab_size'], config['hidden_size'])
        self.wpe = nn.Embedding(config['max_position_embeddings'], config['hidden_size'])
        self.drop = nn.Dropout(0.1)
        self.blocks = nn.ModuleList([DecoderBlock(config) for _ in range(config['num_layers'])])
        self.ln = nn.LayerNorm(config['hidden_size'])
        self.fc = nn.Linear(config['hidden_size'], config['vocab_size'], bias=False)
        self.apply(self._init_weights)
        # weight-tying
        self.wte.weight = self.fc.weight
        print("number of parameters: %.2fM" % (self.get_num_params()/1e6,))
    
    def forward(self, x, y=None):
        device = x.device
        b_sz, t_sz = x.shape
        pos = torch.arange(0, t_sz, device=x.device) # T
        token_emb = self.wte(x) # B, T, H
        pos_emb = self.wpe(pos) # T, H
        x = self.drop(token_emb + pos_emb) # B, T, H
        for block in self.blocks:
            x = block(x)
        x = self.ln(x)
        if y is not None:
            logits = self.fc(x)
            shift_logits = logits[..., :-1, :].contiguous()
            shift_y = y[..., 1:].contiguous() # Need to shift labels by 1 as we are trying to predict next token
            # Need to ignore pad token id 50256 or else model will learn to only predict padding tokens
            loss = F.cross_entropy(shift_logits.view(-1, shift_logits.size(-1)), shift_y.view(-1), ignore_index=50256)
            loss = loss.mean()
        else:
            # (B, T, H) -> (B, 1, V)
            logits = self.fc(x[:, [-1], :])
            loss = None
        return logits, loss

    @torch.no_grad() 
    def generate(self, idx, max_length, pad_token_id=50256, temperature=1.0, top_k=None):
        # idx: B, T
        for _ in range(max_length):
            idx = idx[:, -self.config['window_size']:]
            logits, _ = self(idx)
            logits = logits[:, -1, :] / temperature
            if top_k is not None:
                v, k = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float('-inf')
            probs = F.softmax(logits, dim=-1)
            next_idx = torch.multinomial(probs, num_samples=1)
            idx = torch.cat([idx, next_idx], dim=1)
        return idx
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def get_num_params(self, non_embedding=True):
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n_params -= self.wpe.weight.numel()
        return n_params

class DecoderBlock(nn.Module):
    # Decoder block consisting of attention and mlp sub-blocks 
    # Decoder block interweaves sub-blocks with residual paths and layer norms
    def __init__(self, config):
        super(DecoderBlock, self).__init__()
        self.ln1 = nn.LayerNorm(config['hidden_size'])
        self.attn = MultiheadAttention(config)
        self.ln2 = nn.LayerNorm(config['hidden_size'])
        self.ffn = MLP(config)

    def forward(self, x):
        residual = x
        x = self.ln1(x)
        x = self.attn(x)
        x = residual + x
        residual = x
        x = self.ln2(x)
        x = self.ffn(x)
        x = residual + x
        return x

class MultiheadAttention(nn.Module):
    # Attention sub-block of decoder block
    def __init__(self, config):
        super(MultiheadAttention, self).__init__()
        mx_pos = config['max_position_embeddings']
        self.register_buffer("bias",
            torch.tril(torch.ones((mx_pos,mx_pos), dtype=torch.bool)).view(1, 1, mx_pos, mx_pos),
            persistent=False,
        )
        self.embed_dim = config['hidden_size']
        self.num_heads = config['num_heads']
        self.head_dim = self.embed_dim // self.num_heads
        self.split_size = self.embed_dim
        self.c_attn = Conv1D(3 * self.embed_dim, self.embed_dim)
        self.c_proj = Conv1D(self.embed_dim, self.embed_dim)
        self.attn_dropout = nn.Dropout(0.1)
        self.resid_dropout = nn.Dropout(0.1)
   
    def _split_heads(self, tensor, num_heads, attn_head_size):
        new_shape = tensor.size()[:-1] + (num_heads, attn_head_size)
        tensor = tensor.view(new_shape)
        return tensor.permute(0, 2, 1, 3)  # B, NH, T, E

    def _merge_heads(self, tensor, num_heads, attn_head_size):
        tensor = tensor.permute(0, 2, 1, 3).contiguous()
        new_shape = tensor.size()[:-2] + (num_heads * attn_head_size,)
        return tensor.view(new_shape) # B, T, H

    def _attn(self, query, key, value, attention_mask=None):
        attn_weights = torch.matmul(query, key.transpose(-1, -2)) # (B, NH, T, E) x (B, NH, E, T) -> (B, NH, T, T)
        query_length, key_length = query.size(-2), key.size(-2)
        # attend only to previous tokens
        causal_mask = self.bias[:, :, key_length - query_length : key_length, :key_length]
        # next 3 lines add -inf to the parts where causal mask is 0 (upper diagonal) so that softmax discards future tokens
        mask_value = torch.finfo(attn_weights.dtype).min # -inf
        # creates a scalar tensor with value -inf
        mask_value = torch.full([], mask_value, dtype=attn_weights.dtype, device=attn_weights.device)
        attn_weights = torch.where(causal_mask, attn_weights.to(attn_weights.dtype), mask_value)
        attn_weights = nn.functional.softmax(attn_weights, dim=-1)
        attn_weights = self.attn_dropout(attn_weights)
        attn_output = torch.matmul(attn_weights, value) # (B, NH, T, T) x (B, NH, T, E) -> (B, NH, T, E)
        return attn_output, attn_weights

    def forward(self, x, attention_mask=None):
        # x: B, T, H
        query, key, value = self.c_attn(x).split(self.split_size, dim=2) # q, k, h: B, T, H
        query = self._split_heads(query, self.num_heads, self.head_dim) # B, NH, T, E
        key = self._split_heads(key, self.num_heads, self.head_dim)
        value = self._split_heads(value, self.num_heads, self.head_dim)
        attn_output, attn_weights = self._attn(query, key, value, attention_mask) # B, NH, T, E
        attn_output = self._merge_heads(attn_output, self.num_heads, self.head_dim) # B, T, H
        attn_output = self.c_proj(attn_output) # B, T, H
        attn_output = self.resid_dropout(attn_output)
        return attn_output

class MLP(nn.Module):
    # Feedforward NN sub-block of decoder block
    def __init__(self, config):
        super().__init__()
        embed_dim = config['hidden_size']
        intermediate_size = 4 * embed_dim
        self.c_fc = Conv1D(intermediate_size, embed_dim)
        self.c_proj = Conv1D(embed_dim, intermediate_size)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(0.1)

    def forward(self, hidden_states):
        hidden_states = self.c_fc(hidden_states)
        hidden_states = self.act(hidden_states)
        hidden_states = self.c_proj(hidden_states)
        hidden_states = self.dropout(hidden_states)
        return hidden_states

class Conv1D(nn.Module):
    # huggingface gpt2 implementation uses this as an alternative to nn.Linear
    def __init__(self, out_dim, in_dim):
        super().__init__()
        self.out_dim = out_dim
        self.weight = nn.Parameter(torch.empty(in_dim, out_dim))
        self.bias = nn.Parameter(torch.zeros(out_dim))
        nn.init.normal_(self.weight, std=0.02)

    def forward(self, x):
        size_out = x.size()[:-1] + (self.out_dim,)
        # matmul then add bias
        x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        x = x.view(size_out)
        return x