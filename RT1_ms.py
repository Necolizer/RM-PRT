import numpy as np
import mindspore as ms
from mindspore import Parameter, Tensor
import mindspore.nn as nn
import mindspore.numpy as mnp
import mindspore.ops as ops

from typing import List, Optional, Callable, Tuple
from beartype import beartype

from classifier_free_guidance_ms import *

ms.set_context(device_target="GPU")

def exists(val):
    return val is not None

def default(val, d):
    return val if exists(val) else d

def cast_tuple(val, length = 1):
    return val if isinstance(val, tuple) else ((val,) * length)

# sinusoidal positions

def posemb_sincos_1d(seq, dim, temperature = 10000,  dtype = ms.float32):
    n = mnp.arange(seq)
    omega = mnp.arange(dim // 2) / (dim // 2 - 1)
    omega = 1. / (temperature ** omega)

    n = n[:, None] * omega[None, :]
    pos_emb = ops.Concat(1)((ops.Sin()(n), ops.Cos()(n)))
    return pos_emb.astype(dtype)
# helper classes

class Residual(nn.Cell):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def construct(self, x):
        return self.fn(x) + x

class LayerNorm(nn.Cell):
    def __init__(self, dim):
        super().__init__()
        self.layerNorm = nn.LayerNorm((dim,))

    def construct(self, x):
        return self.layerNorm(x)
    
class FeedForward(nn.Cell):
    def __init__(self, dim, mult = 4, dropout = 0.):
        super().__init__()
        inner_dim = int(dim * mult)
        self.norm = LayerNorm(dim)

        self.net = nn.SequentialCell([
            nn.Dense(dim, inner_dim),
            nn.GELU(),
            nn.Dropout(1.-dropout),
            nn.Dense(inner_dim, dim),
            nn.Dropout(1.-dropout)
        ])
    def construct(self, x, cond_fn = None, norm = True,onlyNorm=False):
        if norm:
            x = self.norm(x)
            if onlyNorm:
                return x


        if cond_fn is not None:
            # adaptive layernorm
            x = cond_fn(x)

        return self.net(x)

class LambdaLayer(nn.Cell):
    def __init__(self, lambda_fn):
        super(LambdaLayer, self).__init__()
        self.lambda_fn = lambda_fn

    def construct(self, x):
        return self.lambda_fn(x)

# MBConv

class SqueezeExcitation(nn.Cell):
    def __init__(self, dim, shrinkage_rate = 0.25):
        super().__init__()
        hidden_dim = int(dim * shrinkage_rate)

        def rearrange(x):
            x=ops.ExpandDims()(x,2)
            x=ops.ExpandDims()(x,2)
            return x

        def fn1(x):
            x=ops.ReduceMean()(x,(2,3))
            return x
        
        self.gate = nn.SequentialCell([
            LambdaLayer(fn1),
            nn.Dense(dim, hidden_dim, has_bias = False),
            nn.SiLU(),
            nn.Dense(hidden_dim, dim, has_bias = False),
            nn.Sigmoid(),
            LambdaLayer(rearrange)
        ])

    def construct(self, x):
        return x * self.gate(x)

class MBConvResidual(nn.Cell):
    def __init__(self, fn, dropout = 0.):
        super().__init__()
        self.fn = fn
        self.dropsample = nn.Dropout(1.-dropout)

    def construct(self, x):
        out = self.fn(x)
        out = self.dropsample(out)
        return out + x

def MBConv(
    dim_in,
    dim_out,
    *,
    downsample,
    expansion_rate = 4,
    shrinkage_rate = 0.25,
    dropout = 0.
):
    hidden_dim = int(expansion_rate * dim_out)
    stride = 2 if downsample else 1

    net = nn.SequentialCell([
        nn.Conv2d(dim_in, hidden_dim, 1,pad_mode='pad',has_bias=True),
        nn.BatchNorm2d(hidden_dim),
        nn.GELU(),
        nn.Conv2d(hidden_dim, hidden_dim, 3, stride = stride, padding = 1, group = hidden_dim,pad_mode='pad',has_bias=True),
        nn.BatchNorm2d(hidden_dim),
        nn.GELU(),
        SqueezeExcitation(hidden_dim, shrinkage_rate = shrinkage_rate),
        nn.Conv2d(hidden_dim, dim_out, 1,pad_mode='pad',has_bias=True),
        nn.BatchNorm2d(dim_out)
    ])

    if dim_in == dim_out and not downsample:
        net = MBConvResidual(net, dropout = dropout)

    return net

# attention related classes

class Attention(nn.Cell):
    def __init__(
        self,
        dim,
        dim_head = 32,
        dropout = 0.,
        window_size = 7
    ):
        super().__init__()
        assert (dim % dim_head) == 0, 'dimension should be divisible by dimension per head'

        self.norm = LayerNorm(dim)

        self.heads = dim // dim_head
        self.scale = dim_head ** -0.5

        self.to_qkv = nn.Dense(dim, dim * 3, has_bias = False)

        self.attend = nn.SequentialCell([
            nn.Softmax(axis = -1),
            nn.Dropout(1.-dropout)
        ])

        self.to_out = nn.SequentialCell([
            nn.Dense(dim, dim, has_bias = False),
            nn.Dropout(1.-dropout)
        ])

        # relative positional bias

        self.rel_pos_bias = nn.Embedding((2 * window_size - 1) ** 2, self.heads)

        pos = mnp.arange(window_size)
        grid = ops.Stack()(ops.Meshgrid(indexing='ij')((pos,pos)))
        grid = grid.reshape(grid.shape[0],-1).transpose(1,0)
        
        rel_pos = ops.ExpandDims()(grid,1) - ops.ExpandDims()(grid,0)
        rel_pos += window_size - 1
        rel_pos_indices = (rel_pos * Tensor([2 * window_size - 1, 1])).sum(axis = -1)
        self.rel_pos_indices=Parameter(rel_pos_indices, name="rel_pos_indices", requires_grad=False)

    def construct(self, x):
        (batch, height, width, window_height, window_width, _), h = x.shape, self.heads

        x = self.norm(x)

        # flatten
        x=x.reshape(batch*height*width,window_height*window_width,-1)

        # project for queries, keys, values
        q, k, v = ops.Split(axis=-1, output_num=3)(self.to_qkv(x))

        # split heads
        b,n,hd=q.shape
        q = q.reshape(b,n,h,-1).transpose(0,2,1,3)
        k = k.reshape(b,n,h,-1).transpose(0,2,1,3)
        v = v.reshape(b,n,h,-1).transpose(0,2,1,3)
        # q, k, v = map(lambda t: t.reshape(b,n,h,-1).transpose(0,2,1,3), (q, k, v))

        # scale
        q = q * self.scale

        # sim
        sim = ops.Einsum('b h i d, b h j d -> b h i j')((q,k))

        # add positional bias

        bias = self.rel_pos_bias(self.rel_pos_indices)
        sim = sim + bias.transpose(2,0,1)
        
        # attention

        attn = self.attend(sim)

        # aggregate

        out = ops.Einsum('b h i j, b h j d -> b h i d')((attn,v))
        
        # merge heads
        b,h,w,d=out.shape
        out = out.reshape(b,h,window_height,window_width,d).transpose(0,2,3,1,4).reshape(b,window_height,window_width,-1)

        # combine heads out

        out = self.to_out(out)
        
        return out.reshape(-1,height,width,*out.shape[1:])

        
class MaxViT(nn.Cell):
    def __init__(
        self,
        *,
        num_classes,
        dim,
        depth,
        dim_head = 32,
        dim_conv_stem = None,
        window_size = 7,
        mbconv_expansion_rate = 4,
        mbconv_shrinkage_rate = 0.25,
        dropout = 0.1,
        channels = 3
    ):
        super().__init__()
        assert isinstance(depth, tuple), 'depth needs to be tuple if integers indicating number of transformer blocks at that stage'

        # convolutional stem

        dim_conv_stem = default(dim_conv_stem, dim)

        self.conv_stem = nn.SequentialCell([
            nn.Conv2d(channels, dim_conv_stem, 3, stride = 2, padding = 1,pad_mode='pad',has_bias=True),
            nn.Conv2d(dim_conv_stem, dim_conv_stem, 3, padding = 1,pad_mode='pad',has_bias=True)
        ])

        # variables

        num_stages = len(depth)

        dims = tuple(map(lambda i: (2 ** i) * dim, range(num_stages)))
        dims = (dim_conv_stem, *dims)
        dim_pairs = tuple(zip(dims[:-1], dims[1:]))

        self.layers = nn.CellList([])

        # shorthand for window size for efficient block - grid like attention

        w = window_size

        # iterate through stages

        def fn1(x):
            b,d,xw,yw=x.shape
            x=x.reshape(b,d,xw//w,w,yw//w,w)
            x=x.transpose(0,2,4,3,5,1)
            return x

        def fn2(x):
            b, X, y, w1, w2, d = x.shape
            x=x.transpose(0,5,1,3,2,4)
            x=x.reshape(b,d,X*w1,y*w2)
            return x

        def fn3(x):
            b,d,wx,wy=x.shape
            x=x.reshape(b,d,w,wx//w,w,wy//w)
            x=x.transpose(0,3,5,2,4,1)
            return x

        def fn4(x):
            b,X,y,w1,w2,d=x.shape
            x=x.transpose(0,5,3,1,4,2)
            x=x.reshape(b,d,w1*X,w2*y)
            return x

        cond_hidden_dims = []

        for ind, ((layer_dim_in, layer_dim), layer_depth) in enumerate(zip(dim_pairs, depth)):
            for stage_ind in range(layer_depth):
                is_first = stage_ind == 0
                stage_dim_in = layer_dim_in if is_first else layer_dim

                cond_hidden_dims.append(stage_dim_in)

                block = nn.SequentialCell([
                    MBConv(
                        stage_dim_in,
                        layer_dim,
                        downsample = is_first,
                        expansion_rate = mbconv_expansion_rate,
                        shrinkage_rate = mbconv_shrinkage_rate
                    ),
                    LambdaLayer(fn1),
                    Residual(Attention(dim = layer_dim, dim_head = dim_head, dropout = dropout, window_size = w)),
                    Residual(FeedForward(dim = layer_dim, dropout = dropout)),
                    LambdaLayer(fn2),

                    LambdaLayer(fn3),
                    Residual(Attention(dim = layer_dim, dim_head = dim_head, dropout = dropout, window_size = w)),
                    Residual(FeedForward(dim = layer_dim, dropout = dropout)),
                    LambdaLayer(fn4)
                ])

                self.layers.append(block)

        embed_dim = dims[-1]
        self.embed_dim = dims[-1]

        self.cond_hidden_dims = cond_hidden_dims

        # mlp head out

        self.mlp_head = nn.SequentialCell([
            LambdaLayer(lambda x: ops.ReduceMean()(x,(2,3))),
            LayerNorm(embed_dim),
            nn.Dense(embed_dim, num_classes)
        ])

    @beartype
    def construct(
        self,
        x,
        texts: Optional[List[str]] = None,
        cond_fns: Optional[Tuple[Callable, ...]] = None,
        cond_drop_prob = 0.,
        return_embeddings = False
    ):
        x = self.conv_stem(x)

        if cond_fns is None:
            cond_fns = (None,) * len(self.layers)

        for stage, cond_fn in zip(self.layers, cond_fns):
            if cond_fn is not None:
                x = cond_fn(x)

            x = stage(x)

        if return_embeddings:
            return x

        return self.mlp_head(x)

# attention


class TransformerAttention(nn.Cell):
    def __init__(
        self,
        dim,
        causal = False,
        dim_head = 64,
        dim_context = None,
        heads = 8,
        norm_context = False,
        dropout = 0.1
    ):
        super().__init__()
        self.heads = heads
        self.scale = dim_head ** -0.5
        self.causal = causal
        inner_dim = dim_head * heads

        dim_context = default(dim_context, dim)

        self.norm = LayerNorm(dim)
        self.context_norm = LayerNorm(dim_context) if norm_context else nn.Identity()

        self.attn_dropout = nn.Dropout(1.-dropout)

        self.to_q = nn.Dense(dim, inner_dim, has_bias = False)
        self.to_kv = nn.Dense(dim_context, dim_head * 2, has_bias = False)
        self.to_out = nn.SequentialCell([
            nn.Dense(inner_dim, dim, has_bias = False),
            nn.Dropout(1.-dropout)
        ])

    def construct(
        self,
        x,
        context = None,
        mask = None,
        attn_bias = None,
        attn_mask = None,
        cond_fn: Optional[Callable] = None,
        norm = True,
        onlyNorm = False
    ):
        b = x.shape[0]

        if context is not None:
            context = self.context_norm(context)

        kv_input = default(context, x)

        if norm:
            x = self.norm(x)
            if onlyNorm:
                return x

        if cond_fn is not None:
            # adaptive layer-norm
            x = cond_fn(x)

        q, (k, v) = self.to_q(x), ops.Split(axis=-1, output_num=2)(self.to_kv(kv_input))

        b,n,hd=q.shape
        q=q.reshape(b,n,self.heads,-1).transpose(0,2,1,3)

        q = q * self.scale

        sim = ops.Einsum('b h i d, b j d -> b h i j')((q,k))
        
        if attn_bias is not None:
            sim = sim + attn_bias

        if attn_mask is not None:
            sim = ops.MaskedFill()(sim, ~attn_mask, -np.finfo(np.float32).max)

        if mask is not None:
            mask = ops.ExpandDims()(x,1)
            mask = ops.ExpandDims()(x,1)
            sim = ops.MaskedFill()(sim, ~mask, -np.finfo(np.float32).max)

        if self.causal:
            i, j = sim.shape[-2:]
            causal_mask = mnp.triu(mnp.ones((i, j), dtype = ms.bool_),j - i + 1)
            sim = ops.MaskedFill()(sim, causal_mask, -np.finfo(np.float32).max)

        attn = nn.Softmax(axis=-1)(sim)
        
        attn = self.attn_dropout(attn)

        out = ops.Einsum('b h i j, b j d -> b h i d')((attn, v))
        
        b,h,n,d=out.shape
        out=out.transpose(0,2,1,3).reshape(b,n,-1)
        return self.to_out(out)

@beartype
class Transformer(nn.Cell):
    def __init__(
        self,
        dim,
        dim_head = 64,
        heads = 8,
        depth = 6,
        attn_dropout = 0.,
        ff_dropout = 0.
    ):
        super().__init__()
        self.layers = nn.CellList([])
        for _ in range(depth):
            self.layers.append(nn.CellList([
                TransformerAttention(dim = dim, heads =  heads, dropout = attn_dropout),
                FeedForward(dim = dim, dropout = ff_dropout)
            ]))

    def construct(
        self,
        x,
        cond_fns: Optional[Tuple[Callable, ...]] = None,
        attn_mask = None
    ):
        if cond_fns is None:
            cond_fns = (None,) * len(self.layers * 2)

        cond_fns = iter(cond_fns)

        for attn, ff in self.layers:
             x = attn(x, attn_mask = attn_mask, cond_fn = next(cond_fns)) + x
             x = ff(x, cond_fn = next(cond_fns)) + x
        return x

# token learner module

class TokenLearner(nn.Cell):
    """
    https://arxiv.org/abs/2106.11297
    using the 1.1 version with the MLP (2 dense layers with gelu) for generating attention map
    """

    def __init__(
        self,
        *,
        dim,
        ff_mult = 2,
        num_output_tokens = 8,
        num_layers = 2
    ):
        super().__init__()
        inner_dim = dim * ff_mult * num_output_tokens

        self.num_output_tokens = num_output_tokens
        self.net = nn.SequentialCell([
            nn.Conv2d(dim * num_output_tokens, inner_dim, 1, group = num_output_tokens,pad_mode='pad',has_bias=True),
            nn.GELU(),
            nn.Conv2d(inner_dim, num_output_tokens, 1, group = num_output_tokens,pad_mode='pad',has_bias=True),
        ])

    def construct(self, x):
        b,_,c,h,w=x.shape
        x=x.reshape(-1,c,h,w)
        shape=[1]*len(x.shape)
        shape[1]*=self.num_output_tokens
        x=mnp.tile(x,shape)
        attn = self.net(x)

        attn=ops.ExpandDims()(attn,1)

        g=self.num_output_tokens
        b,gc,h,w=x.shape
        x=x.reshape(b,g,gc//g,h,w).transpose(0,2,1,3,4)

        x = ops.ReduceMean()(x * attn,(3,4))
        x = x.reshape(b,_,*x.shape[1:])
        return x
    
# Robotic Transformer

@beartype
class RT1(nn.Cell):
    def __init__(
        self,
        *,
        vit: MaxViT,
        num_actions = 11,
        action_bins = 256,
        depth = 6,
        heads = 8,
        dim_head = 64,
        token_learner_ff_mult = 2,
        token_learner_num_layers = 2,
        token_learner_num_output_tokens = 8,
        cond_drop_prob = 0.2,
        use_attn_conditioner = False,
        conditioner_kwargs: dict = dict(),
        use_mask=False
    ):
        super().__init__()
        self.vit = vit

        self.num_vit_stages = len(vit.cond_hidden_dims)

        conditioner_klass = TextConditioner
    
        self.conditioner = conditioner_klass(
            hidden_dims = (*tuple(vit.cond_hidden_dims), *((vit.embed_dim,) * depth * 2)),
            hiddens_channel_first = (*((True,) * self.num_vit_stages), *((False,) * depth * 2)),
            cond_drop_prob = cond_drop_prob,
            **conditioner_kwargs
        )
        self.token_learner = TokenLearner(
            dim = vit.embed_dim,
            ff_mult = token_learner_ff_mult,
            num_output_tokens = token_learner_num_output_tokens,
            num_layers = token_learner_num_layers
        )

        self.num_learned_tokens = token_learner_num_output_tokens

        self.transformer_depth = depth

        self.transformer = Transformer(
            dim = vit.embed_dim,
            dim_head = dim_head,
            heads = heads,
            depth = depth
        )

        self.cond_drop_prob = cond_drop_prob
        self.to_logits = nn.SequentialCell([
            LayerNorm(vit.embed_dim),
        ])
            
    @classifier_free_guidance
    def construct(
        self,
        video,
        texts=None,
        text_embeds = None,
        cond_drop_prob = 0.
    ):
        depth = self.transformer_depth
        if cond_drop_prob is None:
            cond_drop_prob=self.cond_drop_prob
        # cond_drop_prob = default(cond_drop_prob, self.cond_drop_prob)

        frames = video.shape[2]
        
        repeat_batch = tuple([frames,] * self.num_vit_stages+[1,] * self.transformer_depth * 2)
        
        text_embeds = self.conditioner(
            text_embeds=text_embeds,
            cond_drop_prob = cond_drop_prob,
            repeat_batch = tuple([frames,] * self.num_vit_stages+[1,] * self.transformer_depth * 2)
        )
        # conditioner forward
        repeat_batch = repeat_batch if isinstance(repeat_batch, tuple) else ((repeat_batch,) * self.conditioner.num_condition_fns)

        
        cond_fns = []
        return_cond_text_embeds=[]
        
        for cond, cond_hiddens_channel_first, cond_repeat_batch in zip(self.conditioner.conditioners, self.conditioner.hiddens_channel_first, repeat_batch):
            cond_text_embeds=text_embeds.repeat(cond_repeat_batch,axis=0)
            return_cond_text_embeds.append(cond_text_embeds)
            cond_fns.append(cond)
        cond_index=0

        vit_cond_fns, transformer_cond_fns = cond_fns[:-(depth * 2)], cond_fns[-(depth * 2):]
        vit_cond_text_embeds, transformer_cond_text_embeds = return_cond_text_embeds[:-(depth * 2)], return_cond_text_embeds[-(depth * 2):]

        video = video.transpose(0,2,1,3,4)
        b,f,c,h,w=video.shape
        images=video.reshape(-1,c,h,w)

        #vit forward
        x = images
        x = self.vit.conv_stem(x)
        
        for stage, cond_fn,cond_text_embeds in zip(self.vit.layers, vit_cond_fns,vit_cond_text_embeds):
            if cond_fn is not None:
                shape=x.shape
                x=x.reshape(shape[0],shape[1],-1)
                x=x.transpose(0,2,1)
                x = self.conditioner.conditioners[cond_index](return_cond_text_embeds[cond_index],x)
                cond_index+=1
                x=x.transpose(0,2,1)
                x=x.reshape(shape)
            x = stage(x)
        tokens = x
        

        tokens=tokens.reshape([b,f]+list(tokens.shape[1:]))
        
        learned_tokens = self.token_learner(tokens)
        
        learned_tokens=learned_tokens.transpose(0,1,3,2)
        b,f,n,c=learned_tokens.shape
        learned_tokens=learned_tokens.reshape(b,f*n,c)
        
        # causal attention mask

        attn_mask = mnp.triu(mnp.ones((frames, frames), dtype = ms.bool_),1)
        attn_mask=attn_mask.astype(ms.int32).repeat(self.num_learned_tokens,axis=0).repeat(self.num_learned_tokens,axis=1).astype(ms.bool_)
        
        # sinusoidal positional embedding
        seq=frames
        dim=learned_tokens.shape[-1]
        temperature=10000
        n = mnp.arange(seq)
        omega = mnp.arange(dim // 2) / (dim // 2 - 1)
        omega = 1. / (temperature ** omega)

        n = n[:, None] * omega[None, :]
        pos_emb = ops.Concat(1)((ops.Sin()(n), ops.Cos()(n)))
        learned_tokens = learned_tokens + pos_emb.repeat(self.num_learned_tokens,axis=0)
        
        # attention
        # transformer forward
        x=learned_tokens
        attn_mask = ~attn_mask

        for i in range(self.transformer_depth):
            x = self.transformer.layers[i][0](x,onlyNorm=True)
            shape=x.shape
            x=x.reshape(shape[0],-1,shape[-1])
            x = self.conditioner.conditioners[cond_index](return_cond_text_embeds[cond_index],x)
            cond_index+=1
            x=x.reshape(shape)
            x = self.transformer.layers[i][0](x, attn_mask = attn_mask, cond_fn = None, norm = False) + x

            x = self.transformer.layers[i][1](x,onlyNorm=True)
            shape=x.shape
            x=x.reshape(shape[0],-1,shape[-1])
            x = self.conditioner.conditioners[cond_index](return_cond_text_embeds[cond_index],x)
            cond_index+=1
            x=x.reshape(shape)
            x = self.transformer.layers[i][1](x, cond_fn = None, norm = False) + x

        attended_tokens = x


        b,fn,d=attended_tokens.shape
        pooled=attended_tokens.reshape(b,frames,-1,d)
        pooled=ops.ReduceMean()(pooled,2)
        logits = self.to_logits(pooled).squeeze(1)
        return logits
