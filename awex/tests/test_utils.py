# Licensed to the Awex developers under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.

import torch
from transformers import GPT2Config


class SimpleGPT(torch.nn.Module):
    def __init__(
        self,
        vocab_size=50257,
        n_embd=768,
        n_layer=12,
        n_head=12,
        block_size=1024,
        dropout=0.1,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.n_embd = n_embd
        self.block_size = block_size

        # Token and position embeddings
        self.token_embedding = torch.nn.Embedding(vocab_size, n_embd)
        self.position_embedding = torch.nn.Embedding(block_size, n_embd)
        self.dropout = torch.nn.Dropout(dropout)

        # Transformer blocks
        self.blocks = torch.nn.ModuleList(
            [TransformerBlock(n_embd, n_head, dropout) for _ in range(n_layer)]
        )

        # Final layer norm and output projection
        self.ln_f = torch.nn.LayerNorm(n_embd)
        self.lm_head = torch.nn.Linear(n_embd, vocab_size, bias=False)

        # Initialize weights randomly
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, torch.nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, torch.nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, torch.nn.LayerNorm):
            torch.nn.init.zeros_(module.bias)
            torch.nn.init.ones_(module.weight)

    def forward(self, idx, targets=None):
        B, T = idx.shape

        # Token embeddings + position embeddings
        tok_emb = self.token_embedding(idx)  # (B, T, n_embd)
        pos_emb = self.position_embedding(
            torch.arange(T, device=idx.device)
        )  # (T, n_embd)
        x = self.dropout(tok_emb + pos_emb)

        # Transformer blocks
        for block in self.blocks:
            x = block(x)

        # Final layer norm and logits
        x = self.ln_f(x)
        logits = self.lm_head(x)  # (B, T, vocab_size)

        loss = None
        if targets is not None:
            loss = torch.nn.functional.cross_entropy(
                logits.view(-1, logits.size(-1)), targets.view(-1)
            )

        return logits, loss


class TransformerBlock(torch.nn.Module):
    def __init__(self, n_embd, n_head, dropout):
        super().__init__()
        self.ln1 = torch.nn.LayerNorm(n_embd)
        self.attn = MultiHeadAttention(n_embd, n_head, dropout)
        self.ln2 = torch.nn.LayerNorm(n_embd)
        self.mlp = MLP(n_embd, dropout)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class MultiHeadAttention(torch.nn.Module):
    def __init__(self, n_embd, n_head, dropout):
        super().__init__()
        assert n_embd % n_head == 0
        self.n_head = n_head
        self.n_embd = n_embd
        self.head_dim = n_embd // n_head

        # QKV projection
        self.c_attn = torch.nn.Linear(n_embd, 3 * n_embd)
        self.c_proj = torch.nn.Linear(n_embd, n_embd)
        self.attn_dropout = torch.nn.Dropout(dropout)
        self.resid_dropout = torch.nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.shape

        # Calculate query, key, values
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.n_embd, dim=2)

        # Reshape for multi-head attention
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)  # (B, nh, T, hs)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)  # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)  # (B, nh, T, hs)

        # Attention
        att = (q @ k.transpose(-2, -1)) * (1.0 / (self.head_dim**0.5))
        att = torch.nn.functional.softmax(att, dim=-1)
        att = self.attn_dropout(att)
        y = att @ v  # (B, nh, T, hs)

        # Reshape and project
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.resid_dropout(self.c_proj(y))
        return y


class MLP(torch.nn.Module):
    def __init__(self, n_embd, dropout):
        super().__init__()
        self.c_fc = torch.nn.Linear(n_embd, 4 * n_embd)
        self.gelu = torch.nn.GELU()
        self.c_proj = torch.nn.Linear(4 * n_embd, n_embd)
        self.dropout = torch.nn.Dropout(dropout)

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x


def simple_torch_model():
    torch.manual_seed(42)

    # Model hyperparameters
    vocab_size = 50257
    n_embd = 768
    n_layer = 12
    n_head = 12
    block_size = 1024
    dropout = 0.1

    # Create the model
    model = SimpleGPT(
        vocab_size=vocab_size,
        n_embd=n_embd,
        n_layer=n_layer,
        n_head=n_head,
        block_size=block_size,
        dropout=dropout,
    )

    # Create a matching Hugging Face config
    config = GPT2Config(
        vocab_size=vocab_size,
        n_positions=block_size,  # max_position_embeddings
        n_embd=n_embd,  # hidden_size
        n_layer=n_layer,  # num_hidden_layers
        n_head=n_head,  # num_attention_heads
        n_inner=4 * n_embd,  # intermediate_size (MLP expansion)
        activation_function="gelu",
        resid_pdrop=dropout,
        embd_pdrop=dropout,
        attn_pdrop=dropout,
        layer_norm_epsilon=1e-5,
        initializer_range=0.02,
        bos_token_id=50256,
        eos_token_id=50256,
    )

    return model, config
