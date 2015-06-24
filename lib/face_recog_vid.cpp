#include "opencv2/core/core.hpp"
#include "opencv2/contrib/contrib.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/objdetect/objdetect.hpp"

#include <iostream>
#include <fstream>
#include <sstream>

using namespace cv;
using namespace std;

static void read_csv(const string& filename, vector<Mat>& images, vector<int>& labels, char separator = ';') {
    std::ifstream file(filename.c_str(), ifstream::in);
    if (!file) {
        string error_message = "No valid input file was given, please check the given filename.";
        CV_Error(CV_StsBadArg, error_message);
    }
    string line, path, classlabel;
    while (getline(file, line)) {
        stringstream liness(line);
        getline(liness, path, separator);
        getline(liness, classlabel);
        if(!path.empty() && !classlabel.empty()) {
            images.push_back(imread(path, 0));
            labels.push_back(atoi(classlabel.c_str()));
        }
    }
}

int main(int argc, const char *argv[]) {
    if (argc != 5) {
        cout << "usage: " << argv[0] << " </path/to/haar_cascade> </path/to/csv.ext> </path/to/device id>" << endl;
        cout << "\t </path/to/haar_cascade> -- Path to the Haar Cascade for face detection." << endl;
        cout << "\t </path/to/haar_cascade> -- Path to the Haar Cascade for eye detection." << endl;
        cout << "\t </path/to/csv.ext> -- Path to the CSV file with the face database." << endl;
        cout << "\t <device id> -- The webcam device id to grab frames from." << endl;
        exit(1);
    }
    string face_haar = string(argv[1]);
    string eye_haar = string(argv[2]);
    string fn_csv = string(argv[3]);
    int deviceId = atoi(argv[4]);
    vector<Mat> images;
    vector<int> labels;
    try {
        read_csv(fn_csv, images, labels);
    } catch (cv::Exception& e) {
        cerr << "Error opening file \"" << fn_csv << "\". Reason: " << e.msg << endl;
        exit(1);
    }
    int im_width = images[0].cols;
    int im_height = images[0].rows;

    Ptr<FaceRecognizer> model = createFisherFaceRecognizer();
    model->train(images, labels);

    CascadeClassifier haar_cascade_face, haar_cascade_eyes;
    haar_cascade_face.load(face_haar);
    haar_cascade_eyes.load(eye_haar);
    VideoCapture cap(deviceId);
    if(!cap.isOpened()) {
        cerr << "Capture Device ID " << deviceId << "cannot be opened." << endl;
        return -1;
    }
    Mat frame, frame_1, frame_2, frame_3;
    for(;;) {
        cap >> frame;
	//Denoising to avoid Flickers
	frame_1 = frame;
	frame_2 = frame_1;
	frame_3 = frame_2;
	
        Mat original = frame.clone();
        Mat original_1 = frame_1.clone();
        Mat original_2 = frame_2.clone();
        Mat original_3 = frame_3.clone();

        Mat gray, gray_1, gray_2, gray_3;
        cvtColor(original, gray, CV_BGR2GRAY);
        cvtColor(original_1, gray_1, CV_BGR2GRAY);
        cvtColor(original_2, gray_2, CV_BGR2GRAY);
        cvtColor(original_3, gray_3, CV_BGR2GRAY);
        vector< Rect_<int> > faces;
 	vector< Rect_<int> > eyes, eyes_1, eyes_2, eyes_3;
        haar_cascade_face.detectMultiScale(gray, faces, 1.1, 2, 0);
        haar_cascade_eyes.detectMultiScale(gray, eyes);
        haar_cascade_eyes.detectMultiScale(gray_1, eyes_1);
        haar_cascade_eyes.detectMultiScale(gray_2, eyes_2);
        haar_cascade_eyes.detectMultiScale(gray_3, eyes_3);
	
        for(int i = 0; i < faces.size(); i++) {
            Rect face_i = faces[i];
            Mat face = gray(face_i);
            Mat face_resized;
            cv::resize(face, face_resized, Size(im_width, im_height), 1.0, 1.0, INTER_CUBIC);
            int prediction_face = model->predict(face_resized);
            rectangle(original, face_i, CV_RGB(0, 255, 0), 1);
            string box_text;
	    if (prediction_face == 21)
	        box_text = format("Prediction = %s", "Vignesh");
	    else if (prediction_face == 26)
		box_text = format("Prediction = %s", "Brijesh");
	    else
                box_text = format("Prediction = %d", prediction_face);
            int pos_x = std::max(face_i.tl().x - 10, 0);
            int pos_y = std::max(face_i.tl().y - 10, 0);
            putText(original, box_text, Point(pos_x, pos_y), FONT_HERSHEY_PLAIN, 1.0, CV_RGB(0,255,0), 2.0);
        }
	for (int i = 0; i < eyes.size(); i++) {
	    Rect eyes_i = eyes[i];
	    Rect eyes_i_1 = eyes_1[i];
	    Rect eyes_i_2 = eyes_2[i];
	    Rect eyes_i_3 = eyes_3[i];
	    //Mat eye = gray(eyes_i);
	    //Mat eyes_resized;
            //cv::resize(eye, eyes_resized, Size(im_width, im_height), 1.0, 1.0, INTER_CUBIC);
            //int prediction_eyes = model->predict(eyes_resized);
            rectangle(original, eyes_i, CV_RGB(0, 0, 255), 1);
            rectangle(original, eyes_i_1, CV_RGB(0, 0, 127), 1);
            rectangle(original, eyes_i_2, CV_RGB(0, 0, 63), 1);
            rectangle(original, eyes_i_3, CV_RGB(0, 0, 31), 1);
	}
        imshow("Detecting Faces....", original);
        char key = (char) waitKey(20);
        if(key == 27)
            break;
    }
    return 0;
}
