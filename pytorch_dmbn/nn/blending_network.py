#!/usr/bin/python
# -*- coding: utf-8 -*-
"""Blending Network

* 

Reference:
    * [Deep-Modality-Blending-Networks](https://github.com/magic-sword/Deep-Modality-Blending-Networks/blob/main/notebooks/DMBN.ipynb)
    * [GlobalAveragePooling1D layer](https://keras.io/keras_core/api/layers/pooling_layers/global_average_pooling1d/)
    * [PyTorchでGlobal Max PoolingとGlobal Average Poolingを使う方法](https://www.xn--ebkc7kqd.com/entry/pytorch-pooling)
    * [TORCH.MULTIPLY](https://pytorch.org/docs/stable/generated/torch.multiply.html)
    * [TORCH.ADD](https://pytorch.org/docs/stable/generated/torch.add.html)
    * [TORCH.CONCATENATE](https://pytorch.org/docs/stable/generated/torch.concatenate.html)

Todo:
    * 

"""

import torch
import torch.nn as nn

from .time_distributed import TimeDistributed

class BlendingEncoder(nn.Module):
    def __init__(
            self,
            encoders = []
            ):
        """BlendingNetwork

        Conv2d、MaxPool2dを繰り返し実行し、画像をエンコードする層

        Args:
            image_size (int,int): 入力画像サイズ

        Shape:
            Input: :math:`(N, C_{in}, H_{in}, W_{in})`
            Output: :math:`(N, C_{out})`
            , where

        .. math::
            N = バッチデータ数
            C_{in} = 画像データのチャンネル数(channels[0])
            H_{in} = 画像データの高さ(image_size[0])
            W_{in} = 画像データの幅(image_size[1])
            C_{out} = 最終出力の全結合層の特徴数(inear_features)

        Examples:
            Consider a batch of 32 Image samples, where each sample is a 128x128 RGB image with channels_first data format,
            The batch input shape is (32, 3, 128, 128).
            You can then use ImageEncoder:

            >>> inputs = torch.rand(32, 3, 128, 128)
            >>> image_encoder = ImageEncoder(
                    image_size = (128,128),
                    channels = [3,32,64,64,128,128,256], 
                    conv_kernel_size = 3,
                    pool_kernel_size = 2,
                    linear_features = 64 
                )
            >>> outputs = image_encoder(inputs)
            >>> outputs.shape
                torch.Size([32, 64])

        Note:
            注意事項などを記載

        """
        super().__init__()
        self.encoders = nn.ModuleList(encoders)

    def forward(self, inputs=[], coefficients=[], target=0):
        # エンコーダーを実行
        input_len = len(inputs)
        encords = []
        for i in range(input_len):
            x = inputs[i]
            x = self.encoders[i](x)

            # Multiply coefficient
            coefficient = 1
            if (len(coefficients) > i):
                    coefficient = coefficients[i]
            x = torch.multiply(x, coefficient)
            
            # エンコード結果を格納
            encords.append(x)

        general_representation = encords[0]
        for i in range(1, input_len):
            general_representation = torch.add(general_representation, encords[i]) 

        # concatenate target time
        target_shape = general_representation.shape[:-1] + (1,)
        target_tensor = torch.full(target_shape, target)
        x = torch.concatenate((general_representation, target_tensor), dim=-1)

        return x