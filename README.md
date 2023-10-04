# PyTorch-Deep-Modality-Blending-Networks
Amazing Models (Deep-Modality-Blending-Networks) package for PyTorch

# Reference

* [[1]](https://www.sciencedirect.com/science/article/pii/S0893608021004329?via%3Dihub) Imitation and mirror systems in robots through Deep Modality Blending Networks
* [[2]](https://qiita.com/studio_haneya/items/9aad8f9ede11e58b41a8) Pythonパッケージの作り方
* [[3]](https://qiita.com/c60evaporator/items/e1ecccab07a607487dcf) 【PyPI 】Pythonの自作ライブラリをpipに公開する方法
* [[4]](https://www.sphinx-doc.org/ja/master/usage/extensions/napoleon.html) sphinx.ext.napoleon -- NumPy および Google スタイルの docstring をドキュメントに取り込む

# Pip Upload
## Build library
以下のコマンドを実施し、ソースコード配布物をビルドします
<pre>
python setup.py sdist
</pre>

ライブラリのルートフォルダ（setup.pyがあるフォルダ）で以下のコマンドを実施し、Wheelパッケージをビルドします
<pre>
python setup.py bdist_wheel
</pre>

## Upload to Test Pip
PyPIは一度アップロードすると同じバージョンで再アップロードできない（再アップロードにはバージョンを上げる必要がある）
本番用PyPIでミスをして無駄にバージョンが上がることを防ぐため、事前にテスト用PyPIで問題なくアップロードできることを確認することが望ましいです。

以下のコマンドで、テスト用PyPIへライブラリをアップロードします。
<pre>
python -m twine upload --repository testpypi dist/*
</pre>

テスト用PyPIからpipでライブラリインストールを確認
<pre>
pip install --index-url https://test.pypi.org/simple/pytorch_dmbn
pip uninstall pytorch_dmbn
</pre>

## Upload to Pip
本番用PyPIへのライブラリアップロード
<pre>
twine upload --repository pypi dist/*
pip install pytorch_dmbn
</pre>

本番用PyPIからpipでライブラリインストールを確認
<pre>
pip install pytorch_dmbn
</pre>

# Docstring style
pytorchのDocstringを確認したところ、Google styleだった
今回のプロジェクトもpytorchに倣って、Google styleで記載することにする