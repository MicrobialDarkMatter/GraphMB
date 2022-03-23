from setuptools import setup
from setuptools.command.install import install
import os
import subprocess
from distutils.util import convert_path

main_ns = {}
ver_path = convert_path("src/graphmb/version.py")
with open(ver_path) as ver_file:
    exec(ver_file.read(), main_ns)

setup(
    name="graphmb",
    version=main_ns["__version__"],
    packages=["graphmb"],
    package_dir={"": "src"},
    setup_requires=["setuptools~=58.0", "wheel"],
    install_requires=[
        "wheel",
        "requests",
        "setuptools>57.5.0" 'importlib; python_version == "3.7"',
        "vamb @ git+git://github.com/AndreLamurias/vamb",
        "networkx==2.6.2",
        # torch==1.7.1
        "scikit-learn==0.24.2",
        "dgl==0.6.1",
        "tqdm==4.61.2",
        "pandas==1.3.5",
        "tensorflow==2.4.0",
        "matplotlib==3.5.1",
        "rich==11.0.0",
    ],
    entry_points={
        "console_scripts": ["graphmb=graphmb.main:main"],
    },
)
