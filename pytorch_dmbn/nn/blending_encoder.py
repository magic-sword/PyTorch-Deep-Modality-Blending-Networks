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
            encoders = [],
            in_features = 129,
            linear_features = 1024,
            ):
        """BlendingEncoder

        複数のエンコード層を実行し、特徴量を1つにブレンドする
        エンコーダーの特徴量は、全て加算する
        その後、特徴量を1つ増やして、出力のターゲット時間の値を入れる
        coefficientsの値を0にすることで、認識のエンコーダーからの入力をマスクする

        Args:
            encoders (list): nn.Moduleエンコーダーのリスト
            in_features (int): エンコーダーの出力特徴数
            linear_features (int): 最終出力で実行する全結合層の特徴数

        Shape:
            Inputs: :math:`[(N, L1_{in}), (N, L2_{in}), ...]`
            Output: :math:`(N, C_{out}+1)`
            , where

        .. math::
            N = バッチデータ数
            L1_{in} = 1つめのエンコーダーの入力
            L_{out} = 最終出力の全結合層の特徴数(inear_features)

        Examples:
            >>> encoders = [
                ImageEncoder(image_size = (128,128), channels = [3,32,64,64,128,128,256], linear_features = 128),
                LinearEncoder(channels = [8,32,64,64,128,128,256,128])
            ]
            >>> encoders = [TimeAverageEncoder(encoder) for encoder in encoders]
            >>> m = BlendingEncoder(encoders, in_features=129, linear_features=1024,)

            >>> inputs = [
                torch.rand(32, 10, 3, 128, 128),
                torch.rand(32, 10, 8)
            ]
            >>> output = m(inputs=inputs, coefficients=[1,0], target=0.1)
            >>> print(output.shape)
                torch.Size([32, 1024])

        Note:
            * エンコーダーから出力形は、バッチ数、時間ステップ、チャンネルの3次元である必要がある
                * (N, T, C)
                * ここで、チャンネルの数は全てのエンコーダで統一すること
            * in_featuresは、エンコーダの出力チャンネルに、ターゲット時間を追加した値となる
                * in_features = C + 1

        """
        super().__init__()
        self.encoders = nn.ModuleList(encoders)
        
        # 全結合層
        self.linear = nn.Linear(in_features=in_features, out_features=linear_features)
        self.relu = nn.ReLU()

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

        # 全結合層
        x = self.linear(x)
        x = self.relu(x)

        return x