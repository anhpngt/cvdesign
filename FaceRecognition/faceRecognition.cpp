#include <iostream>
#include <vector>

#include <opencv2/core/core.hpp>
#include <opencv2/objdetect/objdetect.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

using namespace std;

int main(int argc, char** argv)
{
  // Loading the Face Detection model
  cv::CascadeClassifier face_detector;
  face_detector.load("/home/echo/cvdesign/FaceRecognition/haarcascades_models/haarcascade_frontalface_alt2.xml");

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
      cv::rectangle(frame, faces[i], cv::Scalar(0, 255, 0), 2, 8, 0);
      cv::putText(frame, "unknown", faces[i].tl(), cv::FONT_HERSHEY_PLAIN, 2, cv::Scalar::all(255), 2, 8);
    }

    // Recognition on detected faces
    

    // Display
    cv::imshow("image", frame);
    int key = cv::waitKey(10);
    if(key == 'q') break;
  }

  return(0);
}