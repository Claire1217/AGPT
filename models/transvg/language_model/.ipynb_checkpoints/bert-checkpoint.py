# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Backbone modules.
"""
from collections import OrderedDict

import torch
import torch.nn.functional as F

from torch import nn
from typing import Dict, List

from utils.misc import NestedTensor, is_main_process
# from .position_encoding import build_position_encoding

# from pytorch_pretrained_bert.modeling import BertModel
# from transformers import BertModel
from transformers import AutoTokenizer, AutoModel


class BERT(nn.Module):
    def __init__(self, args, name: str, train_bert: bool, hidden_dim: int, max_len: int, enc_num, max_query_len):
        super().__init__()
        # if name == 'bert-base-uncased' :
        #     self.num_channels = 768
        # else:
        #     self.num_channels = 1024
        self.args = args
        self.num_channels = 768
        self.enc_num = enc_num
        self.tokenizer = AutoTokenizer.from_pretrained(name)
        self.bert = AutoModel.from_pretrained(name)
        self.max_query_len = max_query_len
        
        if not train_bert:
            for parameter in self.bert.parameters():
                parameter.requires_grad_(False)

    def forward(self, text):
        text_inputs = self.tokenizer(text, return_tensors="pt", max_length=self.max_query_len, truncation=True, padding="max_length").to(self.args.device)
        text_ids = text_inputs['input_ids']
        text_masks = text_inputs['attention_mask']
        if self.enc_num > 0:
            # # pytorch_pretrained_bert version
            # all_encoder_layers, _ = self.bert(tensor_list.tensors, token_type_ids=None, attention_mask=tensor_list.mask)
            # # use the output of the X-th transformer encoder layers
            # xs = all_encoder_layers[self.enc_num - 1]

            # transformers bert version
            bert_output = self.bert(text_ids, attention_mask=text_masks)
            xs = bert_output.last_hidden_state
        else:
            
            xs = self.bert.embeddings.word_embeddings(text_ids)

        mask = text_masks.to(torch.bool)
        mask = ~mask
        out = NestedTensor(xs, mask)

        return out

def build_bert(args):
    # position_embedding = build_position_encoding(args)
    train_bert = args.lr_bert > 0
    bert = BERT(args, args.bert_model, train_bert, args.hidden_dim, args.max_query_len, args.bert_enc_num, args.max_query_len)
    # model = Joiner(bert, position_embedding)
    # model.num_channels = bert.num_channels 
    return bert
