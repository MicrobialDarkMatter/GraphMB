from setuptools import setup
from setuptools.command.install import install
import os
import subprocess


setup(
    name="graphmb",
    version="0.1.0",
    packages=["graphmb"],
    package_dir={"": "src"},
    # cmdclass={"install": InstallLocalPackage},
    install_requires=[
        "wheel",
        "requests",
        'importlib; python_version == "3.7"',
        "vamb @ git+https://github.com/AndreLamurias/vamb",
        "networkx==2.6.2",
        # torch==1.7.1
        "scikit-learn==0.24.2",
        "dgl==0.6.1",
        "tqdm==4.61.2",
    ],
)
