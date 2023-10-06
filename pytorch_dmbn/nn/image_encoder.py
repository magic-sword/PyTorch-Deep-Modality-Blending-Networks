#!/usr/bin/python
# -*- coding: utf-8 -*-
"""ImageEncoder

* 

Reference:
    * [Deep-Modality-Blending-Networks](https://github.com/magic-sword/Deep-Modality-Blending-Networks/blob/main/notebooks/DMBN.ipynb)

Todo:
    * 

"""

import torch.nn as nn
import torch.nn.functional as F

class ImageEncoder(nn.Module):
    def __init__(
            self,
            image_size = (128,128),
            channels = [3,32,64,64,128,128,256], 
            conv_kernel_size = 3,
            pool_kernel_size = 2,
            linear_features = 128 
            ):
        """ImageEncoder

        Conv2d、MaxPool2dを繰り返し実行し、画像をエンコードする層

        Args:
            image_size (int,int): 入力画像サイズ
            channels ([int]): Conv2dのチャンネルサイズと、繰り返し回数
            conv_kernel_size (int): Conv2dのkernel_size
            pool_kernel_size (int): MaxPool2dのkernel_size
            inear_features (int): 最終出力で実行する全結合層の特徴数

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
        
        image_size_h, image_size_w = image_size
        layers = []
        channels_len = len(channels)
        for i in range(channels_len -1):
            in_channels = channels[i]
            out_channels = channels[i+1]
            
            # エンコードレイヤー
            layers.append(nn.Conv2d(in_channels, out_channels, conv_kernel_size, padding='same'))
            layers.append(nn.ReLU())
            layers.append(nn.MaxPool2d(pool_kernel_size))
            
            # pooling後の画像サイズ
            image_size_h = image_size_h / pool_kernel_size
            image_size_w = image_size_w / pool_kernel_size
        self.encoder_layers = nn.ModuleList(layers)
        
        
        self.flatten = nn.Flatten()
        in_features= int(channels[-1] * image_size_h * image_size_w)
        self.linear = nn.Linear(in_features=in_features, out_features=linear_features)
        self.relu = nn.ReLU()

    def forward(self, x):
        for l in self.encoder_layers:
            x = l(x)
        x = self.flatten(x)
        x = self.linear(x)
        x = self.relu(x)
        return x