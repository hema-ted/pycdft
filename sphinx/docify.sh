#!/usr/bin/env bash

# clean and build sphinx documentation for pycdft
# also installed:
#   pip install --user Sphinx 
#   python3 -m pip install nbsphinx --user
#   pip install sphinx_rtd_theme --user
#   

# CHANGE THIS
home_dir="/home/wwwennie/bin2/pycdft/sphinx/"

cd $home_dir
rm -r $home_dir/build/*

sphinx-build -b html $home_dir/source/ $home_dir/build/
