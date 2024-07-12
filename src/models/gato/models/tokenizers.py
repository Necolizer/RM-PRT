import torch
import torch.nn as nn
from typing import Union, Dict, Any


def mu_law_encode(x, mu=100, m=256):
    # Appendix B. Agent Data Tokenization Details
    sign = torch.sign(x)
    numerator = torch.log(torch.abs(x) * mu + 1.0)
    denominator = torch.log(m * mu + 1.0)
    return (numerator / denominator) * sign


def tokenize_continuous_values(x, mu=100, m=256, bins=1024, shift=None):
    # Appendix B. Agent Data Tokenization Details
    # > Finally, they are discretized using bins of uniform width on the domain [-1, 1].
    c = mu_law_encode(x, mu, m)

    # > We use 1024 bins and shift the resulting integers
    # > so they are not overlapping with the ones used for text tokens.
    c = (c + 1) * (bins / 2)
    c = c.int()
    if shift is not None:
        c += shift
    return c


class ContinuousValueTokenizer(nn.Module):
    def __init__(self,
                 config: Union[GatoConfig, Dict[str, Any]],
                 mu=100, m=256, bins=1024,
                 **kwargs):
        super(ContinuousValueTokenizer, self).__init__()
        if isinstance(config, dict):
            config = GatoConfig(**config)
        self.config = config
        self.mu = mu
        self.m = m
        self.bins = bins

    def forward(self, inputs):
        return tokenize_continuous_values(inputs, self.mu, self.m, self.bins, shift=self.config.vocabulary_size)

    def get_config(self):
        return super(ContinuousValueTokenizer, self).get_config()
