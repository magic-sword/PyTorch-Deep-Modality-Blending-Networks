#!/usr/bin/python
# -*- coding: utf-8 -*-
"""Linear Encoder

* 

Reference:
    * [Deep-Modality-Blending-Networks](https://github.com/magic-sword/Deep-Modality-Blending-Networks/blob/main/notebooks/DMBN.ipynb)

Todo:
    * 

"""

import torch.nn as nn

class LinearEncoder(nn.Module):
    def __init__(
        self,
        channels = [8,32,64,64,128,128,256,128]
    ):
        """LinearEncoder

        Linearを繰り返し実行し、エンコードする層

        Args:
            channels ([int]): Linearのチャンネルサイズと、繰り返し回数

        Shape:
            Input: :math:`(N, C_{in})`
            Output: :math:`(N, C_{out})`
            , where

        .. math::
            N = バッチデータ数
            C_{in} = 入力の特徴数(channels[0])
            C_{out} = 出力の特徴数(channels[-1])

        Examples:
            >>> inputs = torch.rand(32, 8)
            >>> encoder = LinearEncoder(
                    channels = [8,32,64,64,128,128,256,128]
                )
            >>> outputs = encoder(inputs)
            >>> outputs.shape
                torch.Size([32, 128])

        Note:
            * 
        """
        super().__init__()
        channels_len = len(channels)
        layers = []
        for i in range(channels_len -1):
            in_channels = channels[i]
            out_channels = channels[i+1]
            layers.append(nn.Linear(in_features=in_channels, out_features=out_channels))
            layers.append(nn.ReLU())
        self.layers = nn.ModuleList(layers)

    def forward(self, x):
        for l in self.layers:
            x = l(x)
        return x