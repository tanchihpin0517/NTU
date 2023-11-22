import torch.nn as nn
from transformers import GPTNeoXForCausalLM, GPTNeoXConfig

class RemiTransformer(nn.Module):
    def __init__(self, hp):
        super().__init__()
        model_config = GPTNeoXConfig(
            vocab_size = hp.vocab_size,
            hidden_size = hp.d_model,
            num_hidden_layers = hp.num_layers,
            num_attention_heads = hp.num_heads,
            intermediate_size = hp.d_feedforward,
            hidden_act = hp.activation,
            hidden_dropout = hp.dropout,
            max_position_embeddings = hp.max_seq_len+1,
        )
        self.transformer = GPTNeoXForCausalLM(model_config)

    def forward(self, *args, **kwargs):
        return self.transformer(*args, **kwargs)

