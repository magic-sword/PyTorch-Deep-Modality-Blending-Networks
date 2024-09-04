import os
import re
import torch
import torch.nn as nn


class NumberingSaver():
    def __init__(self, root:str='./save/dmbn/', digit:int=8):
        """NumberingSaver

        番号をつけて保存、読み込みをするクラス

        Args:
            root (str): 保存先のディレクトリパス
            digit (int): ファイル名を0埋めする桁数

        Examples:
            >>> from pytorch_dmbn.utils.save import NumberingSaver
            >>> saver = NumberingSaver()
            >>> if(saver.is_exist_file('fast_name')):
                    model, blending_encoder_num = saver.load_latest('fast_name')
            >>> saver.save(model, 'fast_name', 100)

        Note:
            * 

        """
        self.root = root
        self.digit = digit
        self.delimiter = '_'
        self.extension = '.pth'
        self.file_name_expression = re.compile(f"(.+){self.delimiter}(\d+){self.extension}") # ファイル名の正規表現

        # データのルートフォルダがなかったら生成する
        os.makedirs(self.root, exist_ok=True)
    
    def __get_file_name(self, name:str, num:int):
        return name + self.delimiter + str(num).zfill(self.digit) + '.pth'
    
    def __serch_valid_numbers(self, name:str):
        """

        有効なファイル名のパターンを検索し、数字を取得する
        """
        file_names = os.listdir(self.root)
        nums = []
        for file_name in file_names:
            m = self.file_name_expression.match(file_name)
            # ファイル名が正規表現に一致しない場合はスキップ
            if(m == None):
                continue
            m_groups = m.groups()
            # 名前が不一致の場合はスキップ
            if(name != m_groups[0]):
                continue
            nums.append(m_groups[1])
        return nums

    def save(self, obj, name:str, num:int):
        """save

        保存

        Args:
            obj (Object): 保存対象のオブジェクト
            name (str): 保存ファイルの名称
            num (int): 保存する番号(エポック数など)

        """
        file_name = self.__get_file_name(name, num)
        file_path = os.path.join(self.root, file_name)
        torch.save(obj, file_path)

    def load(self, name:str, num:int):
        """load

        読み込み

        Args:
            name (str): 保存ファイルの名称
            num (int): 保存する番号(エポック数など)

        Returns:
            Any: torch.loadで読み込んだオブジェクト
        """
        file_name = self.__get_file_name(name, num)
        file_path = os.path.join(self.root, file_name)

        return torch.load(file_path)
    
    def load_latest(self, name:str):
        """load_latest

        番号の一番高いファイルを読み込み

        Args:
            name (str): 保存ファイルの名称

        Returns:
            Any: torch.loadで読み込んだオブジェクト
            int: 読み込んだ際の番号
        """
        nums = self.__serch_valid_numbers(name)
        
        # ファイル名のパターンが一致するファイルが存在しない場合はエラー
        if(len(nums) < 1):
            raise FileNotFoundError(f"There are no files in the root directory.: {self.root}")
        
        nums.sort()
        latest_num = nums[-1]
        print(f"load latest number: {latest_num}")
        return self.load(name, latest_num), int(latest_num)

    def is_exist_file(self, name:str):
        """is_exist_file

        保存されたファイルの存在を調べる

        Args:
            name (str): 保存ファイルの名称

        Returns:
            bool: True,有効なファイルが存在する False,有効なファイルが存在しない
        """
        nums = self.__serch_valid_numbers(name)
        if(len(nums) < 1):
            return False
        return True