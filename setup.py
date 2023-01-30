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
    setup_requires=["setuptools~=58.0", "wheel", "sphinx-rtd-theme", "twine"],
    install_requires=[
        "wheel",
        "requests",
        "networkx==2.6.2",
        "torch==1.12.1",
        "scikit-learn==0.24.2",
        "tqdm==4.61.2",
        "pandas==1.3.5",
        "tensorflow==2.11.0",
        "tqdm==4.61.2",
        "numpy==1.23.5",
        "mlflow==2.1.1"

    ],
    entry_points={
        "console_scripts": ["graphmb=graphmb.main:main"],
    },
)
