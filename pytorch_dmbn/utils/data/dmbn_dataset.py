import os
import pandas as pd
import math
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision.io import read_image
import pathlib
import urllib.request
import shutil
import zipfile

class DMBNDataset(Dataset):
    def __init__(self, root='./data/dmbn_dataset/', download=True, obs_max=5):
        self.root = root
        self.obs_max = obs_max
        self.data_id_max = 50
        self.action_types = ['move','grasp']

        self.rand_obs_num()
        if(download):
            self.__download()


    def rand_obs_num(self):
        """ 観測数の乱数再設定

        DMBNでは、データセット内の時系列データからサンプリングするデータの個数(観測数)をランダムに決定している
        
        データの読み出し1毎に観測数がランダムだと、テンソルの形状が異なってミニバッチにまとめることができなくなる
        ミニバッチの読み出しの間で観測数を統一するため、乱数の再設定をこの関数で管理する

        トレーニングのイテレート毎にこの関数を呼び出すことで、統一した観測数でデータを読み出し、ミニバッチを作成できるようになる
        """
        self.obs_num = np.random.randint(0,self.obs_max)+1
        return self.obs_num

    def __len__(self):
        return self.data_id_max * len(self.action_types)

    def __getitem__(self, idx):
        data_id = str(math.floor(idx / len(self.action_types)))
        action_type = self.action_types[idx % len(self.action_types)]

        pose_path = os.path.join(self.root, data_id, action_type, f'joint_{data_id}.txt')
        pose = torch.tensor(np.loadtxt(pose_path), dtype=torch.get_default_dtype())
        pose[:,-1] *= 10 # ?

        time_len = pose.shape[0]
        times = torch.linspace(0,1,time_len)
        perm = torch.randperm(time_len)

        n = self.obs_num
        observation = torch.zeros((n,4,128,128))
        observation_pose = torch.zeros((n,8))
        for i in range(n):
            observation[i,0, :,:] = torch.ones((128,128))*times[perm[i]]  # 画像のチャンネルに時間軸データを追加
            img_path = os.path.join(self.root, data_id, action_type,f'{perm[i]}.jpeg')
            observation[i,1:,:,:] = read_image(img_path)/255. # 画像データ(RBG)を読み込み
            observation_pose[i,0] = times[perm[i]]    # 関節データのチャンネルに時間軸データを追加
            observation_pose[i,1:] = pose[perm[i]]    # 関節データを追加

        target_X = torch.full((1,), times[perm[n]])

        # 画像の正解データ　正解のRBG(3チャンネル)と、各チャンネルに対する分散が0(3チャンネル)を用意する
        target_Y = torch.zeros((6,128,128))
        img_path = os.path.join(self.root, data_id, action_type,f'{perm[i]}.jpeg')
        target_Y[:3,:,:] = read_image(img_path)/255. # 教師データの画像データ(RBG)を読み込み

        # 関節の正解データ　正解の関節値(7チャンネル)と、各チャンネルに対する分散が0(7チャンネル)を用意する
        target_Y_pose = torch.zeros((14))
        target_Y_pose[:7] = pose[perm[n]]

        # 感覚データの係数
        coef = np.random.rand()
        img_coef = torch.ones((128)) * coef
        pose_coef = torch.ones((128)) * (1-coef)

        # 出力時間の形状を、エンコーダの出力形状に合わせる必要がある
        target_time = torch.full((1,), perm[n])

        """
        sample = {
            "observation": observation
            , "observation_pose": observation_pose
            , "target_X": target_X
            , "img_coef": img_coef
            , "pose_coef": pose_coef
            , "target_Y": target_Y
            , "target_Y_pose": target_Y_pose
            , "data_id": data_id
            , "perm[n]": perm[n]
        }
        """
        return (
            observation,
            observation_pose,
            target_X,
            img_coef,
            pose_coef,
            target_Y,
            target_Y_pose,
            data_id,
            target_time
        ) 
    
    def __download(self):
        # データのルートフォルダがなかったら生成する
        os.makedirs(self.root, exist_ok=True)

        # ダウンロード済みフラグファイル
        downloaded_file_path = os.path.join(self.root, 'is_downloaded')
        if os.path.exists(downloaded_file_path):
            # ダウンロード済みファイルが存在するので、処理を中断
            return

        # 参考のgitからgitをダウンロード
        url='https://codeload.github.com/myunusseker/Deep-Modality-Blending-Networks/zip/refs/heads/main'
        save_name=os.path.join(self.root, 'main.zip')
        urllib.request.urlretrieve(url, save_name)

        # zipから解凍
        with zipfile.ZipFile(save_name) as zf:
            zf.extractall(self.root)

        # 解凍したフォルダから、データのルートフォルダへ移動
        src_dir = os.path.join(self.root, 'Deep-Modality-Blending-Networks-main/data/data2020')
        for path in os.listdir(src_dir):
            shutil.move(os.path.join(src_dir, path), self.root)

        # ダウンロードしたzipと解凍したファイルを削除
        os.remove(save_name)
        shutil.rmtree(os.path.join(self.root, 'Deep-Modality-Blending-Networks-main'))

        # ダウンロード済みフラグファイルを作成し、再実行の処理時間を削減する
        touch_file = pathlib.Path(downloaded_file_path)
        touch_file.touch()
