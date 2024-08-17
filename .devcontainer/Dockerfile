# 作成したpytorch  バージョンは最新のものを確認
FROM magicsword0001/pytorch-jupyterlab-gpu:latest
USER $USERNAME

# パッケージの作成に必要
RUN pip install setuptools twine wheel

# 以下のコマンドを実施し、ソースコード配布物をビルドします
# RUN python setup.py sdist

# ライブラリのルートフォルダ（setup.pyがあるフォルダ）で以下のコマンドを実施し、Wheelパッケージをビルドします
# RUN  python setup.py bdist_wheel

# 設定ファイルをホームディレクトリに配置
COPY .devcontainer/.pypirc ~/

# 以下のコマンドで、テスト用PyPIへライブラリをアップロードします。
# RUN python -m twine upload --repository testpypi dist/*


# The description failed to render in the default format of reStructuredText.とエラーが出た時
# README.rstのフォーマット違反により、レンダリングが失敗することによって発生します。
# 原因箇所を特定
RUN pip install readme-renderer
# RUN python -m readme_renderer README.rst -o README.html


# テスト用PyPIからpipでライブラリインストールを確認
# RUN pip install --index-url https://test.pypi.org/simple/pytorch_dmbn
# RUN pip uninstall pytorch_dmbn

# 本番用PyPIへのライブラリアップロード
# RUN twine upload --repository pypi dist/*
# RUN pip install

# MineRLのインストール
RUN sudo apt-get update
RUN sudo apt-get install openjdk-8-jdk -y