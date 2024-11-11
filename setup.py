from setuptools import setup, find_packages

setup(
    name="manga-colorization",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "torch",
        "torchvision",
        "numpy",
        "opencv-python",
        "pandas",
        "scikit-image",
        "tqdm",
        "pyyaml"
    ]
) 