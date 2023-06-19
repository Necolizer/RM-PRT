import pandas as pd

from mindspore import ops
from mindspore import Tensor
from mindspore import context
import mindspore.nn as nn
import mindspore as ms
context.set_context(device_target="GPU")

from RT1_ms import MaxViT, RT1
from utils import *
class Predictor(nn.Cell):
    def __init__(self,RT1,extractors,mlp_extractor,action_net):
        super().__init__()
        self.database=pd.read_csv('instructions/training.csv')
        self.jointsArrange=initJointsArrange()
        self.RT1=RT1
        self.extractors=extractors
        self.mlp_extractor=mlp_extractor
        self.action_net=action_net

    def predict(self,obs):
        head_rgb=obs['head_rgb']
        state=obs['state']
        head_rgb=Tensor(head_rgb)
        state=Tensor(state)
        head_rgb=ops.ExpandDims()(head_rgb,0)
        state=ops.ExpandDims()(state,0)
        head_rgb=head_rgb.transpose(0,3,1,2)
        head_rgb=ops.ExpandDims()(head_rgb,2)

        instructionID=obs['instruction'][0]
        instruction=self.database[self.database['id']==instructionID]['instruction'].values[0]
        instructions = [instruction]
        text_embeds=self.RT1.conditioner.embed_texts(instructions)
        train_logits = self.RT1(head_rgb, text_embeds=text_embeds)
        fea1=self.extractors[0](train_logits)
        state[:,:3]/=2000
        state[:,3:3+21]/=Tensor(self.jointsArrange[:,1])
        state[:,3+21:3+21+3]/=100
        state=ops.Concat(1)((state[:,3:3+7],state[:,-7-4:-4],state[:,-1].reshape(-1,1)))
        fea2=self.extractors[1](state)
        encoded_tensor=ops.Concat(1)((fea1,fea2))
        mlp_feature=self.mlp_extractor(encoded_tensor)
        action=self.action_net(mlp_feature)
        return ops.clip_by_value(action, -1., 1.).asnumpy()
    
def predictor_construct():
    vit = MaxViT(
        num_classes = 1000,
        dim_conv_stem = 64,
        dim = 96,
        dim_head = 32,
        depth = (2, 2, 5, 2),
        window_size = 7,
        mbconv_expansion_rate = 4,
        mbconv_shrinkage_rate = 0.25,
        dropout = 0.1,
        channels=4
    )
    model = RT1(
        vit = vit,
        num_actions = 8,
        action_bins = 128,
        depth = 6,
        heads = 8,
        dim_head = 64,
        cond_drop_prob = 0.2,
    )
    fc1 = nn.SequentialCell([nn.Dense(768, 256), nn.ReLU()])
    fc2 = nn.Dense(15, 64)
    extractors = nn.SequentialCell([fc1,fc2])
    mlp_extractor = nn.SequentialCell([nn.Dense(320, 256),nn.Tanh(),nn.Dense(256, 128),nn.Tanh()])
    action_net = nn.Dense(128, 9)

    predictor=Predictor(model,extractors,mlp_extractor,action_net)
    ms.load_checkpoint('predictor.ckpt',net=predictor)
    return predictor