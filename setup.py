#! /usr/bin/env python
#
# Copyright (C) Hideto Mori


DESCRIPTION = "patchwork for matplotlib"
LONG_DESCRIPTION = ""

DISTNAME         = 'patchworklib'
MAINTAINER       = 'Hideto Mori'
MAINTAINER_EMAIL = 'morityunasfc.keio.ac.jp/hidto7592agmail.com'
URL              = 'https://github.com/ponnhide/patchworklib'
LICENSE          = 'GNU General Public License v3.0'
DOWNLOAD_URL     = 'https://github.com/ponnhide/patchworklib'
VERSION          = '0.0.0'
PYTHON_REQUIRES  = ">=3.7"

INSTALL_REQUIRES = [
    'matplotlib>=3.2',
    'dill'
]


PACKAGES = [
    'patchworklib'
]

CLASSIFIERS = [
    'Intended Audience :: Science/Research',
    'Programming Language :: Python :: 3.7',
    'Programming Language :: Python :: 3.8',
    'Programming Language :: Python :: 3.9',
    'License :: OSI Approved :: GNU General Public License v3.0',
    'Topic :: Bioinformatics',
    'Operating System :: OS Independent',
]


if __name__ == "__main__":
    from setuptools import setup
    import sys
    if sys.version_info[:2] < (3, 7):
        raise RuntimeError("patchworklib requires python >= 3.7.")

    setup(
        name=DISTNAME,
        author=MAINTAINER,
        author_email=MAINTAINER_EMAIL,
        maintainer=MAINTAINER,
        maintainer_email=MAINTAINER_EMAIL,
        description=DESCRIPTION,
        long_description=LONG_DESCRIPTION,
        license=LICENSE,
        url=URL,
        version=VERSION,
        download_url=DOWNLOAD_URL,
        python_requires=PYTHON_REQUIRES,
        install_requires=INSTALL_REQUIRES,
        packages=PACKAGES,
        classifiers=CLASSIFIERS
    )
