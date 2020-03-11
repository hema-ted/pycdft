from setuptools import setup

# make sure PYTHONPATH contains the path of a directory you can write to
# e.g., in this instance
#    export PYTHONPATH="${PYTHONPATH}:/path/to/pycdft/lib/python3.6/site-packages"
#    python setup.py install --prefix="/path/to/pycdft/"
# the "/lib/python3.6/site-packages/" gets tacked on later

setup(
    name="pycdft",
    version="0.9",
    author="He Ma, Wennie Wang, Siyoung Kim, Man Hin Cheng, Marco Govoni, Giulia Galli",
    author_email="mahe@uchicago.edu",
    description="",
    packages=['pycdft'],
    install_requires=[
        "ase",
        "numpy",
        "scipy",
        "pyFFTW",
        "lxml"
    ],
)

