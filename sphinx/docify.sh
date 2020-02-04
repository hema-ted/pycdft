#!/usr/bin/env bash

# clean and build sphinx documentation for pycdft
# also installed:
#   pip install --user Sphinx 
#   python3 -m pip install nbsphinx --user
#   pip install sphinx_rtd_theme --user
#   

# CHANGE THIS
home_dir="/PATH/TO/pycdft/sphinx/"

cd $home_dir
rm -r $home_dir/sphinx_build/*

sphinx-build -b html $home_dir/source/ $home_dir/sphinx_build/
