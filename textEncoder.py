from typing import List
from beartype import beartype

import numpy as np
import mindspore as ms
from mindspore import Parameter, Tensor
import mindspore.nn as nn
import mindspore.numpy as mnp
import mindspore.ops as ops

from mindformers import BertForPreTraining, BertTokenizer
import mindspore.common.dtype as mstype

from mindformers import AutoModel
from mindformers import AutoTokenizer

# helpers

def exists(val):
    return val is not None

def default(val, d):
    return val if exists(val) else d

def get_model_and_tokenizer(name):
    model = AutoModel.from_pretrained(name)
    tokenizer = AutoTokenizer.from_pretrained(name)

    return model, tokenizer

DEFAULT_NAME='./bert'
MAX_LENGTH=128

class BertAdapter():
    def __init__(
        self,
        name
    ):
        name = default(name, DEFAULT_NAME)
        model, tokenizer = get_model_and_tokenizer(name)

        self.name =  name
        self.model = model
        self.tokenizer = tokenizer

    @beartype
    def embed_text(
        self,
        texts: List[str],
        return_text_encodings = False,
        output_device = None
    ):

        encoded = self.tokenizer.batch_encode_plus(
            texts,
            # return_tensors = "ms",
            padding = "max_length",
            max_length = MAX_LENGTH,
            # truncation = True
        )
        input_ids = encoded['input_ids']
        attention_mask = encoded['attention_mask']
        token_type_ids = encoded['token_type_ids']

        input_ids = Tensor(input_ids, mstype.int32)
        attention_mask = Tensor(attention_mask, mstype.int32)
        token_type_ids = Tensor(token_type_ids, mstype.int32)

        output = self.model.bert(input_ids, attention_mask, token_type_ids)
        encoded_text=output[0]
        attention_mask=attention_mask.astype(ms.bool_)
        encoded_text=ops.MaskedFill()(encoded_text, ~attention_mask[...,None], 0.)
        numer = encoded_text.sum(axis = -2)
        denom = attention_mask.sum(axis = -1)[..., None]
        numer=ops.MaskedFill()(numer, denom == 0, 0.)
        mean_encodings = numer / (denom+1e-3)

        return mean_encodings