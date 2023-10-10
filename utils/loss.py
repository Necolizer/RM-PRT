import torch.nn.functional as F
import torch.nn as nn

# 依照RT-1原文： a standard categorical cross-entropy entropy objective
# 但没有实现原文提到的 causal masking

def get_loss_func(loss_func, loss_args):
    if loss_func == 'CrossEntropy':
        loss = nn.CrossEntropyLoss()
    elif loss_func == 'MSELoss':
        loss = nn.MSELoss()
    else:
        print('Loss Not Included')
        loss = None
    
    return loss
