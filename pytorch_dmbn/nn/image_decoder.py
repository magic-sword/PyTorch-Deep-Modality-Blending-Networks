#!/usr/bin/python
# -*- coding: utf-8 -*-
"""ImageDecoder layer

* 

Reference:
    * [Deep-Modality-Blending-Networks](https://github.com/magic-sword/Deep-Modality-Blending-Networks/blob/main/notebooks/DMBN.ipynb)
    * [UpSampling2D layer](https://keras.io/api/layers/reshaping_layers/up_sampling2d/)
    * [UPSAMPLE](https://pytorch.org/docs/stable/generated/torch.nn.Upsample.html)
Todo:
    * 

"""

import numpy
import torch.nn as nn

class ImageDecoder(nn.Module):
    def __init__(
        self,
        in_features = 128,
        linear_features = 1024,
        channels = [256,128,128,64,64,32,16,8,3], 
        conv_kernel_size = 3,
        upsample_scale = 2,
        image_size = (128,128)
    ):
        """ImageEncoder

        Conv2d、MaxPool2dを繰り返し実行し、画像をデコードする層

        Args:
            in_features (int): 入力特徴数
            inear_features (int): 入力に対して実行する全結合層の出力特徴数
            channels ([int]): Conv2dのチャンネルサイズと、繰り返し回数
            conv_kernel_size (int): Conv2dのkernel_size
            upsample_scale (int): Upsampleのscale_factor
            image_size (int,int): 出力目標の画像サイズ

        Shape:
            Input: :math:`(N, C_{in})`
            Output: :math:`(N, C_{out}, H_{in}, W_{in})`
            , where

        .. math::
            N = バッチデータ数
            C_{in} = 入力の特徴数
            C_{out} = 出力画像のチャンネル数(channels[-1])
            H_{out} = 出力画像データの高さ(image_size[0])
            W_{out} = 出力画像データの幅(image_size[1])

        Examples:
            >>> inputs = torch.rand(32, 128)
            >>> image_decoder = ImageEncoder(
                    in_features = 128,
                    linear_features = 1024,
                    channels = [256,128,128,64,64,32,16,8,3], 
                    conv_kernel_size = 3,
                    upsample_scale = 2,
                    image_size = (128,128)
                )
            >>> outputs = image_decoder(inputs)
            >>> outputs.shape
                torch.Size([32, 3, 128, 128])

        Note:
            * 引数のimage_sizeは、画像サイズを保証するものではない
                * upsample_scaleの大きさと、channelsで指定する層の数が不十分だと、画像サイズを十分に大きくできない
                * image_sizeがupsample_scaleの累乗でなければ、画像サイズがジャストにならない

        """
        super().__init__()
        
        # 出力画像の縦横比と、入力特徴数から、初期の画像サイズを逆算する
        image_size_h, image_size_w = image_size
        height_rate = image_size_h / image_size_w # 出力画像の縦横比
        initial_image_area = linear_features / channels[0] # 初期の画像面積
        initial_width = numpy.sqrt(initial_image_area / height_rate) # 初期の画像の横幅
        initial_height =  initial_image_area / initial_width # 初期の画像の高さ
        self.initial_image_size = (channels[0], int(initial_height), int(initial_width)) # 入力特徴数を初期の形に変形
        
        # 全結合層
        self.linear = nn.Linear(in_features=in_features, out_features=linear_features)
        self.relu = nn.ReLU()
        
        # デコードレイヤー層を構築する
        size_h = initial_height
        size_w = initial_width
        channels_len = len(channels)
        layers = []
        for i in range(channels_len -1):
            in_channels = channels[i]
            out_channels = channels[i+1]
            
            # デコードレイヤー
            layers.append(nn.Conv2d(in_channels, out_channels, conv_kernel_size, padding='same'))
            layers.append(nn.ReLU())
            
            # 画像サイズが出力画像サイズに達していれば、アップサンプリングはしない
            if(size_h >= image_size_h and size_h >=image_size_w):
                continue
            
            # アップサンプリングを実行し、画像サイズをスケールアップする
            layers.append(nn.Upsample(scale_factor=upsample_scale))
            # アップサンプリング後の画像サイズ
            size_h = size_h * upsample_scale
            size_w = size_w * upsample_scale
        self.decoder_layers = nn.ModuleList(layers)

    def forward(self, x):
        x = self.linear(x)
        x = self.relu(x)
        
        # 入力特徴数を初期の形に変形
        x = x.reshape(-1, *self.initial_image_size)
        
        # デコードを実行
        for l in self.decoder_layers:
            x = l(x)
        return x