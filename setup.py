from setuptools import setup, find_packages
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
    python_requires=">=3.8",
    package_dir={"": "src"},
    setup_requires=["setuptools~=58.0", "wheel", "sphinx-rtd-theme", "twine"],
    install_requires=[
        "wheel",
        "requests",
        "networkx==2.6.2",
        "torch==1.13.1",
        "tensorflow==2.11.1",
        "tqdm==4.61.2",
        "mlflow==2.2.1",
        "importlib_resources"

    ],
    entry_points={
        "console_scripts": ["graphmb=graphmb.main:main"],
    },
    include_package_data=True,
)
