rm -r run_dir
mkdir run_dir
cd run_dir
../../lib/create_csv.py ../../database/ > my_file.csv
~/.compile_opencv.sh ../../lib/face_recog_vid.cpp
./face_recog_vid ~/OpenCV/data/haarcascades/haarcascade_frontalface_default.xml ~/OpenCV/data/haarcascades/haarcascade_eye.xml my_file.csv 0
