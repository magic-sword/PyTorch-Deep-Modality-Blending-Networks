#!/usr/bin/python
# -*- coding: utf-8 -*-
"""

* 

Reference:
    * [Deep-Modality-Blending-Networks](https://github.com/magic-sword/Deep-Modality-Blending-Networks/blob/main/notebooks/DMBN.ipynb)
    * [Deep-Modality-Blending-Networks](https://github.com/magic-sword/Deep-Modality-Blending-Networks/blob/main/notebooks/DMBN.ipynb)
    * [tf.split](https://www.tensorflow.org/api_docs/python/tf/split)
    * [TORCH.SPLIT])(https://pytorch.org/docs/stable/generated/torch.split.html)
    * [Tensorflow nn.softplus](https://www.geeksforgeeks.org/python-tensorflow-nn-softplus/)
    * [TORCH.NN.FUNCTIONAL.SOFTPLUS](https://pytorch.org/docs/stable/generated/torch.nn.functional.softplus.html)
    * [tfp.distributions.MultivariateNormalDiag](https://www.tensorflow.org/probability/api_docs/python/tfp/distributions/MultivariateNormalDiag)
    * [TensorFlowにおける確率分布の基本的な使い方](https://qiita.com/kozamb/items/dd97093b3d99c3725642)
    * [対角行列](https://ja.wikipedia.org/wiki/%E5%AF%BE%E8%A7%92%E8%A1%8C%E5%88%97)
    * [多変量正規分布の独立性](https://academ-aid.com/statistics/indep-multi-normal)
    * [torch.distributions.multivariate_normal.MultivariateNormal](https://pytorch.org/docs/stable/distributions.html#multivariatenormal)
    * [torch.diag](https://pytorch.org/docs/stable/generated/torch.diag.html)
    * [How to construct a 3D Tensor where every 2D sub tensor is a diagonal matrix in PyTorch?](https://stackoverflow.com/questions/47372508/how-to-construct-a-3d-tensor-where-every-2d-sub-tensor-is-a-diagonal-matrix-in-p)
    * [TORCH.DIAG_EMBED](https://pytorch.org/docs/stable/generated/torch.diag_embed.html)
    * [tf.reduce_meanの使い方と意味](https://qiita.com/maguro27/items/2effbbafc2c8e7a7eb64)
    * [torch.mean](https://pytorch.org/docs/stable/generated/torch.mean.html)
    * [【Pytorch】テンソルを分割する方法(split・chunk)](https://masaki-note.com/2022/05/08/pytorch_split-chunk/)

Todo:
    * 

"""

import torch
import torch.nn as nn

class CNPsLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, outputs, targets):
        '''
        outputs: 予測結果(ネットワークの出力)
        targets: 正解
        '''
        # 予測結果のテンソルを2分割し、多変量正規分布の平均ベクトルと分散・共分散行列にする
        # 分散・共分散行列はdiag()操作により対角行列とし、全ての確率変数が独立となる
        mean, log_sigma = torch.chunk(outputs, 2, dim=1)
        target_value, temp = torch.chunk(targets, 2, dim=1)

        diag_sigma = torch.diag_embed(nn.functional.softplus(log_sigma))
        dist = torch.distributions.multivariate_normal.MultivariateNormal(loc=mean, covariance_matrix=diag_sigma)

        loss = -torch.mean(dist.log_prob(target_value))
        return loss