// https://docs.opencv.org/3.3.1/da/d60/tutorial_face_main.html

#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

using namespace std;

static void read_database(const std::string& file_name, std::vector<cv::Mat>& images, std::vector<int>& labels, char sep = ',')
{
  std::
}

int main(int argc, char** argv)
{
  if(argc < 2)
  {
    cout << "ERROR: Missing input argument." << endl;
    cout << "Usage: ./trainEigenfaces \"CONFIGURATION_FILE.csv\"" << endl;
    return(-1);
  }

  std::vector<cv::Mat> images;
  std::vector<int> labels;
  read_database(argv[1], images, labels);

  // Get all training images
  std::vector<cv::Mat> image_database;
  std::vector<cv::Mat> flattened_data;
  cv::Mat visual;
  for(int i = 1; i <= image_number; i++)
  {
    std::string file_name;
    fs["Image" + std::to_string(i) + ".Location"] >> file_name;
    cv::Mat tmp = cv::imread(file_name);
    if(tmp.cols != 100 || tmp.rows != 100)
    {
      cout << "ERROR: Invalid image size ([" << tmp.cols << ", " << tmp.rows << "])! @ " << file_name << endl;
      return(-1);
    }
    cv::Mat tmp_gray;
    cv::cvtColor(tmp, tmp_gray, cv::COLOR_BGR2GRAY);
    image_database.push_back(tmp_gray);

    // Flatten
    tmp_gray = tmp_gray.reshape(1, 1);
    flattened_data.push_back(tmp_gray);
  }
  cv::Mat fX;
  cv::vconcat(flattened_data, fX);
  fX.convertTo(fX, CV_32F);
  
  cv::hconcat(image_database, visual);
  cv::imshow("img", visual);
  cv::waitKey(-1);

  return 0;
}