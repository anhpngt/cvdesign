#include <iostream>
#include <vector>
#include <string>

#include <opencv2/core/core.hpp>
#include <opencv2/objdetect/objdetect.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/face.hpp>

#define HEIGHT 100
#define WIDTH  100

using namespace std;

int main(int argc, char** argv)
{
  if(argc < 2)
  {
    cout << "ERROR: Missing input database!" << endl;
    cout << "Usage: ./faceRecognition EIGENFACE_DATA.yaml" << endl;
    exit(-1);
  }
  // Loading the Face Detection model
  cv::CascadeClassifier face_detector;
  face_detector.load("/home/echo/cvdesign/FaceRecognition/haarcascades_models/haarcascade_frontalface_alt2.xml");

  // Loading the Face Recognition model
  cout << "Reading database at " << argv[1] << endl;
  cv::Ptr<cv::face::EigenFaceRecognizer> model = cv::face::EigenFaceRecognizer::create();
  model->read("/home/echo/cvdesign/FaceRecognition/eigenfaces_models/eigenfaces_220318.yaml");

  // Loading camera stream
  cv::VideoCapture cap(0);
  if(!cap.isOpened())
  {
    cout << "ERROR: Unable to open camera!" << endl;
    return(-1);
  }

  for(;;)
  {
    // Get frame
    cv::Mat frame;
    cap >> frame;
    if(frame.empty())
    {
      cout << "ERROR: empty image frame!" << endl;
      return(-1);
    }
    cv::Mat frame_src;
    frame.copyTo(frame_src);

    // Detection
    cv::Mat frame_gray;
    cv::cvtColor(frame, frame_gray, cv::COLOR_BGR2GRAY);
    cv::equalizeHist(frame_gray, frame_gray);
    // cv::resize(frame_gray, frame_gray, cv::Size(1280, 960));

    std::vector<cv::Rect> faces;
    face_detector.detectMultiScale(frame_gray, faces, 1.1, 3, 0 | cv::CASCADE_SCALE_IMAGE, cv::Size(100, 100));
    for(size_t i = 0, i_end = faces.size(); i < i_end; i++)
    {
      // Recognition on detected faces
      cv::Mat face_mat = frame_gray(faces[i]);
      cv::resize(face_mat, face_mat, cv::Size(WIDTH, HEIGHT));
      int label_;
      double confidence_;
      std::string face_name_;
      model->predict(face_mat, label_, confidence_);

      switch(label_)
      {
        case 100:
          face_name_ = "Chi Siong"; 
          break;
        case 101: 
          face_name_ = "Kah Yooi";  
          break;
        case 102: 
          face_name_ = "Samuel";    
          break;
        case 103: 
          face_name_ = "Tuan Anh";
          break;
        default: face_name_ = std::to_string(label_);
      }
      std::string label_str = face_name_ + " / " + std::to_string(confidence_);

      

      // Visualize
      cv::rectangle(frame, faces[i], cv::Scalar(0, 255, 0), 2, 8, 0);
      cv::putText(frame, label_str, faces[i].tl(), cv::FONT_HERSHEY_PLAIN, 2, cv::Scalar::all(255), 2, 8);
    }

    // Display
    cv::imshow("image", frame);
    int key = cv::waitKey(10);
    if(key == 'q') break;
  }

  return(0);
}