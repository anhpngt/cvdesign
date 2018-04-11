#include <iostream>
#include <vector>
#include <string>

#include <opencv2/opencv.hpp>

#define HEIGHT 100
#define WIDTH  100

using namespace std;

int main(int argc, char** argv)
{
  // Loading the Face Detection model
  std::string detection_model_file = "/home/echo/cvdesign/FaceRecognition/haarcascades_models/haarcascade_frontalface_alt2.xml";
  cout << "Reading detection model at: " << detection_model_file << endl;
  cv::CascadeClassifier face_detector;
  face_detector.load(detection_model_file);

  // Loading the Face Recognition model: data mean, eigen values/vector/faces
  std::string recognition_model_file = "/home/echo/cvdesign/FaceRecognition/eigenfaces_models/eigenfaces_050418_2.yaml";
  cout << "Reading recognition model at " << recognition_model_file << endl;
  cv::FileStorage efs(recognition_model_file, cv::FileStorage::READ);
  cv::Mat m_labels(0, 0, CV_64FC1);                       // [N x 1]
  cv::Mat m_mean(0, 0, CV_64FC1);                         // [10000 x 1]
  cv::Mat m_eigenvalues(0, 0, CV_64FC1);                  // [200 x 1]
  cv::Mat m_eigenvectors(0, 0, CV_64FC1);                 // [200 x 10000] each eigenvector is stored as row
  cv::Mat m_eigenfaces(0, 0, CV_64FC1);                   // [200 x N] each eigenface is stored as col
  efs["labels"] >> m_labels;
  efs["mean"] >> m_mean;
  efs["eigenvalues"] >> m_eigenvalues;
  efs["eigenvectors"] >> m_eigenvectors;
  efs["eigenfaces"] >> m_eigenfaces;
  efs.release();

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
    // cv::equalizeHist(frame_gray, frame_gray);
    // cv::resize(frame_gray, frame_gray, cv::Size(1280, 960));

    std::vector<cv::Rect> faces;
    face_detector.detectMultiScale(frame_gray, faces, 1.1, 3, 0 | cv::CASCADE_SCALE_IMAGE, cv::Size(40, 40));
    // Recognition on detected faces
    for(size_t i = 0, i_end = faces.size(); i < i_end; i++)
    {
      // Apply PCA
      cv::Mat face_mat = cv::Mat(frame_gray(faces[i]));
      cv::resize(face_mat, face_mat, cv::Size(WIDTH, HEIGHT));
      if(!face_mat.isContinuous())
      {
        cout << "Skipped a discontinuous matrix (" << i << ")" << endl;
        continue;
      }
      cv::Mat m_face = face_mat.reshape(1, 1);
      cv::transpose(m_face, m_face);                      // [10000 x 1] now
      m_face.convertTo(m_face, CV_64FC1);
      m_face = m_face - m_mean;
      cv::Mat m_face_pca = m_eigenvectors * m_face;       // [200 x 1] face under pca representation

      // Do NCC
      int label_;
      double confidence_;
      std::string face_name_;

      cv::Mat m_face_pca_spreaded = cv::repeat(m_face_pca, 1, m_eigenfaces.cols); // spread to [200 x N] for matrix subtraction
      cv::Mat m_ncc = m_eigenfaces - m_face_pca_spreaded; // [200 x N]
      cv::multiply(m_ncc, m_ncc, m_ncc);                  // [200 x N]
      cv::reduce(m_ncc, m_ncc, 0, cv::REDUCE_SUM);        // [1 x N] sum col-wise
      cv::Mat sorted_idx;                                 // [1 x N]
      cv::sortIdx(m_ncc, sorted_idx, cv::SORT_EVERY_ROW | cv::SORT_ASCENDING);
      label_ = m_labels.at<int>(sorted_idx.at<int>(0, 0), 0);
      double score1 = m_ncc.at<double>(0, sorted_idx.at<int>(0, 0));
      double score2 = m_ncc.at<double>(0, sorted_idx.at<int>(0, 1));
      confidence_ = (score2 - score1) / score2;
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
      std::string label_str = face_name_ + " (" + std::to_string(confidence_) + ")";

      // Visualize
      cv::rectangle(frame, faces[i], cv::Scalar(0, 255, 0), 2, 8, 0);
      cv::putText(frame, label_str, faces[i].tl(), cv::FONT_HERSHEY_PLAIN, 2, cv::Scalar::all(255), 2, 8);
    }

    // Display
    cv::imshow("image", frame);
    int key = cv::waitKey(10);
    if(key == 'q')
    {
      cout << "User exit." << endl;
      return(0);
    }
  }

  return(0);
}