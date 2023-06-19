import numpy as np
import mindspore as ms
from mindspore import Parameter, Tensor
import mindspore.nn as nn
import mindspore.numpy as mnp
import mindspore.ops as ops

from functools import wraps, partial


from beartype import beartype
from beartype.typing import Callable, Tuple, Optional, List, Literal, Union
from beartype.door import is_bearable

from inspect import signature

from textEncoder import BertAdapter

# constants

COND_DROP_KEY_NAME = 'cond_drop_prob'

TEXTS_KEY_NAME = 'texts'
TEXT_EMBEDS_KEY_NAME = 'text_embeds'
TEXT_CONDITIONER_NAME = 'text_conditioner'
CONDITION_FUNCTION_KEY_NAME = 'cond_fns'

# helper functions

def exists(val):
    return val is not None

def default(*values):
    for value in values:
        if exists(value):
            return value
    return None

def cast_tuple(val, length = 1):
    return val if isinstance(val, tuple) else ((val,) * length)

# tensor helpers

def prob_mask_like(shape, prob):
    if prob == 1:
        return mnp.ones(shape, dtype = ms.bool_)
    elif prob == 0:
        return mnp.zeros(shape, dtype = ms.bool_)
    else:
        minval = Tensor(0., ms.float32)
        maxval = Tensor(1., ms.float32)
        return ops.uniform(shape,minval,maxval)<prob
    
# classifier free guidance with automatic text conditioning

@beartype
def classifier_free_guidance(
    fn: Callable,
    cond_drop_prob_keyname = COND_DROP_KEY_NAME,
    texts_key_name = TEXTS_KEY_NAME,
    text_embeds_key_name = TEXT_EMBEDS_KEY_NAME,
    cond_fns_keyname = CONDITION_FUNCTION_KEY_NAME,
    text_conditioner_name = TEXT_CONDITIONER_NAME
):
    fn_params = signature(fn).parameters

    auto_handle_text_condition = texts_key_name not in fn_params and text_embeds_key_name not in fn_params
    assert not (auto_handle_text_condition and cond_fns_keyname not in fn_params), f'{cond_fns_keyname} must be in the wrapped function for autohandling texts -> conditioning functions - ex. forward(..., {cond_fns_keyname})'

    @wraps(fn)
    def inner(
        self,
        *args,
        cond_scale: float = 1.,
        rescale_phi: float = 0.,
        **kwargs
    ):
        @wraps(fn)
        def fn_maybe_with_text(self, *args, **kwargs):
            if auto_handle_text_condition:
                texts = kwargs.pop('texts', None)
                text_embeds = kwargs.pop('text_embeds', None)

                assert not (exists(texts) and exists(text_embeds))

                cond_fns = None

                text_conditioner = getattr(self, text_conditioner_name, None)

                # auto convert texts -> conditioning functions

                if exists(texts) ^ exists(text_embeds):

                    assert is_bearable(texts, Optional[List[str]]), f'keyword `{texts_key_name}` must be a list of strings'

                    assert exists(text_conditioner) and is_bearable(text_conditioner, Conditioner), 'text_conditioner must be set on your network with the correct hidden dimensions to be conditioned on'

                    cond_drop_prob = kwargs.pop(cond_drop_prob_keyname, None)

                    text_condition_input = dict(texts = texts) if exists(texts) else dict(text_embeds = text_embeds)

                    cond_fns = text_conditioner(**text_condition_input, cond_drop_prob = cond_drop_prob)

                elif isinstance(text_conditioner, NullConditioner):
                    cond_fns = text_conditioner()

                kwargs.update(cond_fns = cond_fns)

            return fn(self, *args, **kwargs)

        # main classifier free guidance logic

        if self.training:
            assert cond_scale == 1, 'you cannot do condition scaling when in training mode'

            return fn_maybe_with_text(self, *args, **kwargs)

        assert cond_scale >= 1, 'invalid conditioning scale, must be greater or equal to 1'
        
        kwargs_without_cond_dropout = {**kwargs, cond_drop_prob_keyname: 0.}
        kwargs_with_cond_dropout = {**kwargs, cond_drop_prob_keyname: 1.}

        logits = fn_maybe_with_text(self, *args, **kwargs_without_cond_dropout)

        if cond_scale == 1:
            return logits

        null_logits = fn_maybe_with_text(self, *args, **kwargs_with_cond_dropout)
        scaled_logits = null_logits + (logits - null_logits) * cond_scale

        if rescale_phi <= 0:
            return scaled_logits

        # proposed in https://arxiv.org/abs/2305.08891
        # as a way to prevent over-saturation with classifier free guidance
        # works both in pixel as well as latent space as opposed to the solution from imagen

        dims = tuple(range(1, logits.ndim - 1))
        rescaled_logits = scaled_logits * (logits.std(dim = dims, keepdim = True) / scaled_logits.std(dim = dims, keepdim= True))
        return rescaled_logits * rescale_phi + (1. - rescale_phi) * logits

    return inner

# dimension adapters

def rearrange_channel_last(fn):
    @wraps(fn)
    def inner(hiddens):
        shape=hiddens.shape
        hiddens=hiddens.reshape(shape[0],-1,shape[-1])
        conditioned = fn(hiddens)
        conditioned=conditioned.reshape(shape)
        return conditioned
    return inner

def rearrange_channel_first(fn):
    """ will adapt shape of (batch, feature, ...) for conditioning """

    @wraps(fn)
    def inner(hiddens):
        shape=hiddens.shape
        hiddens=hiddens.reshape(shape[0],shape[1],-1)
        hiddens=hiddens.transpose(0,2,1)
        conditioned =  fn(hiddens)
        conditioned=conditioned.transpose(0,2,1)
        conditioned=conditioned.reshape(shape)
        return conditioned

    return inner

# conditioning modules

class FiLM(nn.Cell):
    def __init__(
        self,
        dim,
        hidden_dim
    ):
        super().__init__()
        self.net = nn.SequentialCell([
            nn.Dense(dim, hidden_dim * 4),
            nn.SiLU(),
            nn.Dense(hidden_dim * 4, hidden_dim * 2, weight_init='zero')
        ])

    def construct(self, conditions, hiddens):
        scale, shift = ops.Split(axis=-1, output_num=2)(self.net(conditions))
        assert scale.shape[-1] == hiddens.shape[-1], f'unexpected hidden dimesion {hiddens.shape[-1]} or scale shape {scale.shape[-1]} used for conditioning'
        scale = ops.ExpandDims()(scale,1)
        shift = ops.ExpandDims()(shift,1)
        return hiddens * (scale + 1) + shift
    
# film text conditioning

CONDITION_CONFIG = dict(
    bert = BertAdapter
)

MODEL_TYPES = CONDITION_CONFIG.keys()

class Conditioner(nn.Cell):
    pass

# null conditioner

class Identity(nn.Cell):
    def construct(self, t, *args, **kwargs):
        return t
    
@beartype
class NullConditioner(Conditioner):
    def __init__(
        self,
        *,
        num_null_conditioners: int
    ):
        super().__init__()
        self.cond_fns = tuple(Identity() for _ in range(num_null_conditioners))


    def embed_texts(self, texts: List[str]):
        assert False, 'null conditioner cannot embed text'

    def construct(self, *args, **kwarg):
        return self.cond_fns
    
# text conditioner with FiLM

@beartype
class TextConditioner(Conditioner):
    def __init__(
        self,
        *,
        hidden_dims: Tuple[int, ...],
        model_types = 'bert',
        model_names = None,
        cond_drop_prob = 0.,
        hiddens_channel_first = True,
        text_embed_stem_dim_mult = 2,
        dim_latent = 768
    ):
        super().__init__()
        model_types = cast_tuple(model_types)
        model_names = cast_tuple(model_names, length = len(model_types))

        assert len(model_types) == len(model_names)
        assert all([model_type in MODEL_TYPES for model_type in model_types])

        text_models = []

        for model_type, model_name in zip(model_types, model_names):
            klass = CONDITION_CONFIG.get(model_type)
            model = klass(model_name)
            text_models.append(model)

        self.text_models = text_models
        self.latent_dims = [dim_latent for model in text_models]

        self.conditioners = nn.CellList([])

        self.hidden_dims = hidden_dims
        self.num_condition_fns = len(hidden_dims)
        self.hiddens_channel_first = cast_tuple(hiddens_channel_first, self.num_condition_fns) # whether hiddens to be conditioned is channel first or last

        assert len(self.hiddens_channel_first) == self.num_condition_fns

        self.cond_drop_prob = cond_drop_prob

        total_latent_dim = sum(self.latent_dims)
        mlp_stem_output_dim = total_latent_dim * text_embed_stem_dim_mult

        self.text_embed_stem_mlp = nn.SequentialCell([
            nn.Dense(total_latent_dim, mlp_stem_output_dim),
            nn.SiLU()
        ])

        for hidden_dim in hidden_dims:
            self.conditioners.append(FiLM(mlp_stem_output_dim, hidden_dim))

        self.null_text_embed = Parameter(mnp.randn(total_latent_dim))


    def embed_texts(self, texts: List[str]):

        text_embeds = []
        for text_model in self.text_models:
            text_embed = text_model.embed_text(texts)
            text_embeds.append(text_embed)

        return ops.Concat(-1)(text_embeds)
    
    def construct(
        self,
        text_embeds: Optional[Tensor] = None,
        cond_drop_prob = None,
        repeat_batch = 1,  # for robotic transformer edge case
    ) -> Tuple[Callable, ...]:


        assert exists(text_embeds)

        if self.training:
            cond_drop_prob = default(cond_drop_prob, self.cond_drop_prob)
        else:
            assert exists(cond_drop_prob), 'when not training, cond_drop_prob must be explicitly set'
        
        batch = text_embeds.shape[0]

        
        if cond_drop_prob > 0.:
            prob_keep_mask = prob_mask_like((batch, 1), 1. - cond_drop_prob)
            null_text_embeds = ops.ExpandDims()(self.null_text_embed,0)
            
            text_embeds = mnp.where(
                prob_keep_mask,
                text_embeds,
                null_text_embeds
            )

        # text embed mlp stem, as done in unet conditioning in guided diffusion

        text_embeds = self.text_embed_stem_mlp(text_embeds)
        return text_embeds
        # # prepare the conditioning functions

        # repeat_batch = cast_tuple(repeat_batch, self.num_condition_fns)

        # cond_fns = []
        # return_cond_text_embeds=[]
        # for cond, cond_hiddens_channel_first, cond_repeat_batch in zip(self.conditioners, self.hiddens_channel_first, repeat_batch):
        #     cond_text_embeds=text_embeds.repeat(cond_repeat_batch,axis=0)
        #     cond_fn = partial(cond, cond_text_embeds)

        #     wrapper_fn = rearrange_channel_first if cond_hiddens_channel_first else rearrange_channel_last

        #     cond_fns.append(wrapper_fn(cond_fn))
        #     return_cond_text_embeds.append(cond_text_embeds)

        # return tuple(cond_fns)