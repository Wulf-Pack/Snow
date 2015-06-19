rm -r run_dir
mkdir run_dir
cd run_dir
../../lib/create_csv.py ../../database/ > my_file.csv
~/.compile_opencv.sh ../../lib/read_imgs.cpp
./read_imgs my_file.csv
