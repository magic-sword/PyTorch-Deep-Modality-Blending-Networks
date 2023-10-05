#!/usr/bin/python
# -*- coding: utf-8 -*-
"""TimeDistributed layer (時間分散層)

* PytorchにはTimeDistributed layerが存在しないため、Kerasのレイヤーを参考に実装

Reference:
    * [PyTorchでモデル（ネットワーク）を構築・生成](https://note.nkmk.me/python-pytorch-module-sequential/)
    * [Timedistributed CNN](https://discuss.pytorch.org/t/timedistributed-cnn/51707)
    * [Any PyTorch function can work as Keras’ Timedistributed?](https://discuss.pytorch.org/t/any-pytorch-function-can-work-as-keras-timedistributed/1346)
    * [KerasのTimeDistributedをPyTorchで実装する](https://ameblo.jp/nana-to-tomo/entry-12792601144.html)
    * [Keras TimeDistributed layer](https://keras.io/api/layers/recurrent_layers/time_distributed/)
    * Because TimeDistributed applies the same instance of Conv2D to each of the timestamps, the same set of weights are used at each timestamp.

Todo:
    * 

"""

import torch.nn as nn

class TimeDistributed(nn.Module):
    """TimeDistributed layer (時間分散層)

    This wrapper allows to apply a layer to every temporal slice of an input.
    Every input should be at least 3D, and the dimension of index one of the first input will be considered to be the temporal dimension.

    """

    def __init__(self, layer):
        """TimeDistributed

        This wrapper allows to apply a layer to every temporal slice of an input.

        Args:
            layer (torch.nn.Module): a torch.nn.Module instance.

        Shape:
            Input: :math:`(N, T, *L_{in})`
            Output: :math:`(N, T, *L_{out})`
            , where

        .. math::
            N = バッチデータ数
            T = タイムステップ数
            *L_{in} = TimeDistributedが増幅するレイヤーへ、インプットするデータの形状(可変数)
            *L_{out} = TimeDistributedが増幅するレイヤーから、アウトプットするデータの形状(可変数)

        Examples:
            Consider a batch of 32 video samples, where each sample is a 128x128 RGB image with channels_last data format,across 10 timesteps.
            The batch input shape is (32, 10, 3, 128, 128).
            You can then use TimeDistributed to apply the same Conv2D layer to each of the 10 timesteps, independently:

            >>> inputs = torch.rand(32, 10, 3, 128, 128)
            >>> conv_2d_layer = torch.nn.Conv2d(in_channels=3, out_channels=8, kernel_size=3, padding='same')
            >>> time_distributed_layer = pytorch_dmbn.nn.TimeDistributed(conv_2d_layer)
            >>> outputs = time_distributed_layer(inputs)
            >>> outputs.shape
                torch.Size([32, 10, 8, 128, 128])

        Note:
            注意事項などを記載

        """
        super(TimeDistributed, self).__init__()
        self.layer = layer

    def forward(self, x):
        """
        inputs: Input tensor of shape (batch, time, ...) or nested tensors, and each of which has shape (batch, time, ...).
        """
        input_shape = x.shape # (batch_size, time_step, ...) と仮定
        
        x = x.reshape(-1, *input_shape[2:]) # 時系列データを大きなバッチデータに変換
        x = self.layer(x) # レイヤーの処理を実行
        layer_out_shape = x.shape[1:] # レイヤーから出力されたテンソル形状
        
        x = x.reshape(input_shape[0], -1, *layer_out_shape) # バッチと系列長を戻す
        
        return x