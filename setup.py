# Author: magic-sword <magic.sword.0001@gmail.com>
# Copyright (c) 2023 magic-sword
# License: MIT License

from setuptools import setup
import pytorch_dmbn

DESCRIPTION = "pytorch_dmbn: Deep-Modality-Blending-Networks package for PyTorch"
NAME = 'pytorch_dmbn'
AUTHOR = 'Magic Sword'
AUTHOR_EMAIL = 'magic.sword.0001@gmail.com'
URL = 'https://github.com/magic-sword/PyTorch-Deep-Modality-Blending-Networks'
LICENSE = 'MIT License'
DOWNLOAD_URL = 'https://github.com/magic-sword/PyTorch-Deep-Modality-Blending-Networks'
VERSION = pytorch_dmbn.__version__
PYTHON_REQUIRES = ">=3.8"

INSTALL_REQUIRES = [
    'torch'
]

EXTRAS_REQUIRE = {
    'tutorial': [
    ]
}

PACKAGES = [
    'pytorch_dmbn'
]

CLASSIFIERS = [
]

with open('README.md', 'r') as fp:
    readme = fp.read()
with open('CONTACT.txt', 'r') as fp:
    contacts = fp.read()
long_description = readme + '\n\n' + contacts

setup(name=NAME,
      author=AUTHOR,
      author_email=AUTHOR_EMAIL,
      maintainer=AUTHOR,
      maintainer_email=AUTHOR_EMAIL,
      description=DESCRIPTION,
      long_description=long_description,
      long_description_content_type='text/markdown',    # README ファイルをマークダウン形式でアップロードする
      license=LICENSE,
      url=URL,
      version=VERSION,
      download_url=DOWNLOAD_URL,
      python_requires=PYTHON_REQUIRES,
      install_requires=INSTALL_REQUIRES,
      extras_require=EXTRAS_REQUIRE,
      packages=PACKAGES,
      classifiers=CLASSIFIERS
    )
