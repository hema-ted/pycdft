from setuptools import setup

setup(
    name="pycdft",
    version="1.0",
    author="He Ma, Wennie Wang, Siyoung Kim, Man Hin Cheng, Marco Govoni, Giulia Galli",
    author_email="mahe@uchicago.edu",
    packages=['pycdft'],
    install_requires=[
        "ase",
        "numpy",
        "scipy",
        "pyFFTW",
        "lxml"
    ],
)
