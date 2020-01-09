#!/usr/bin/env bash

home_dir="/home/wwwennie/bin2/pycdft/sphinx/"

cd $home_dir
rm -r $home_dir/build/

sphinx-build -b html $home_dir/source/ $home_dir/build/
scp -r $home_dir/build  wwwennie@205.208.85.116:~/Downloads/
