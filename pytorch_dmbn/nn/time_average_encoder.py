#!/usr/bin/python
# -*- coding: utf-8 -*-
"""TimeAverageEncoder layer

* 

Reference:
    * [Deep-Modality-Blending-Networks](https://github.com/magic-sword/Deep-Modality-Blending-Networks/blob/main/notebooks/DMBN.ipynb)

Todo:
    * エンコーダーの出力の形が何次元でも対応できるようにしたい

"""

import torch.nn as nn
import torch.nn.functional as F

from .time_distributed import TimeDistributed

class TimeAverageEncoder(nn.Module):
    def __init__(self, layer):
        """TimeAverageEncoder

        入力の時間軸に対して、時間分散層を実行し、平均値をとる

        Args:
            layer (torch.nn.Module): a torch.nn.Module instance.

        Shape:
            Input: :math:`(N, T, *L_{in})`
            Output: :math:`(N, L_{out})`
            , where

        .. math::
            N = バッチデータ数
            T = タイムステップ数
            *L_{in} = TimeDistributedが増幅するレイヤーへ、インプットするデータの形状(可変数)
            *L_{out} = TimeDistributedが増幅するレイヤーから、アウトプットするデータの形状

        Examples:
            Consider a batch of 32 video samples, where each sample is a 128x128 RGB image with channels_first data format,across 10 timesteps.
            The batch input shape is (32, 10, 3, 128, 128).
            You can then use TimeDistributed to apply the same Conv2D layer to each of the 10 timesteps, independently:

            >>> inputs = torch.rand(32, 10, 3, 128, 128)
            >>> layer = ImageEncoder(
                    image_size = (128,128),
                    channels = [3,32,64,64,128,128,256], 
                    conv_kernel_size = 3,
                    pool_kernel_size = 2,
                    linear_features = 128
                )
            >>> m = pytorch_dmbn.nn.TimeAverageEncoder(layer)
            >>> outputs = m(inputs)
            >>> outputs.shape
                torch.Size([32, 128])

        Note:
            * Encoderとして指定する層は、出力が1次元の場合にのみ対応していいる

        """
        super().__init__()
        self.layer = TimeDistributed(layer)

    def forward(self, x):
        """
        inputs: Input tensor of shape (batch, time, ...) or nested tensors, and each of which has shape (batch, time, ...).
        """
        x = self.layer(x)
        
        # GlobalAveragePooling1D
        x = F.avg_pool2d(x, kernel_size=(x.shape[-2],1))
        x = x.reshape(-1, *x.shape[2:])
        
        return x