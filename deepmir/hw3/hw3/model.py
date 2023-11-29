import torch.nn as nn
from transformers import GPTNeoXForCausalLM, GPTNeoXConfig
from transformers import GPT2LMHeadModel, GPT2Config

class RemiTransformer(nn.Module):
    def __init__(self, hp):
        super().__init__()
        if hp.get("relative_position", True):
            model_config = GPTNeoXConfig(
                vocab_size = hp.vocab_size,
                hidden_size = hp.d_model,
                num_hidden_layers = hp.num_layers,
                num_attention_heads = hp.num_heads,
                intermediate_size = hp.d_feedforward,
                hidden_act = hp.activation,
                hidden_dropout = hp.dropout,
                max_position_embeddings = hp.max_position_embeddings+1,
            )
            self.transformer = GPTNeoXForCausalLM(model_config)
        else:
            model_config = GPT2Config(
                vocab_size = hp.vocab_size,
                n_positions = hp.max_position_embeddings+1,
                n_embd = hp.d_model,
                n_layer = hp.num_layers,
                n_head = hp.num_heads,
                n_inner = hp.d_feedforward,
                activation_function = hp.activation,
                resid_pdrop = hp.dropout,
            )
            self.transformer = GPT2LMHeadModel(model_config)

    def forward(self, *args, **kwargs):
        return self.transformer(*args, **kwargs)

