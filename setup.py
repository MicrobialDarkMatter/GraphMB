from setuptools import setup
from setuptools.command.install import install
import os
import subprocess


setup(
    name="graphmb",
    version="0.1.2",
    packages=["graphmb"],
    package_dir={"": "src"},
    setup_requires=["setuptools~=58.0", "wheel"],
    install_requires=[
        "wheel",
        "requests",
        "setuptools>57.5.0" 'importlib; python_version == "3.7"',
        # "vamb @ git+git://github.com/AndreLamurias/vamb",
        "networkx==2.6.2",
        # torch==1.7.1
        "scikit-learn==0.24.2",
        "dgl==0.6.1",
        "tqdm==4.61.2",
    ],
    entry_points={
        "console_scripts": ["graphmb=graphmb.main:main"],
    },
)
