// https://docs.opencv.org/3.3.1/da/d60/tutorial_face_main.html

#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/face.hpp>

#define HEIGHT 100
#define WIDTH  100

using namespace std;

static int read_database(const std::string& file_name, std::vector<cv::Mat>& images, std::vector<int>& labels, char sep = ',')
{
  images.clear();
  labels.clear();

  std::ifstream file(file_name, std::ifstream::in);
  if(!file)
  {
    cout << "ERROR: Cannot read file: " << file_name << endl;
    return 0;
  }

  std::string line, path_str, label_str;
  while(getline(file, line))
  {
    std::stringstream liness(line);
    getline(liness, path_str, sep);
    getline(liness, label_str);
    if(!path_str.empty() && !label_str.empty())
    {
      cv::Mat tmp = cv::imread(path_str, CV_LOAD_IMAGE_GRAYSCALE);
      if(tmp.cols != WIDTH || tmp.rows != HEIGHT)
      {
        cout << "WARN: Image of invalid dimension " << tmp.size() << " at " << path_str << endl;
        cv::resize(tmp, tmp, cv::Size(100, 100));
        // return 0;
      }
      images.push_back(tmp);
      labels.push_back(std::stoi(label_str));
    }
    else
    {
      cout << "Invalid data at: " << path_str << endl;
    }
  }

  return 1;
}

int main(int argc, char** argv)
{
  if(argc < 3)
  {
    cout << "ERROR: Missing input argument." << endl;
    cout << "Usage: ./trainEigenfaces IMAGE_INDEX_FILE.csv OUTPUT_DIR" << endl;
    return(-1);
  }

  std::vector<cv::Mat> images;
  std::vector<int> labels;
  if(!read_database(argv[1], images, labels)) return(-1);

  if(images.size() < 2)
  {
    cout << "ERROR: Insufficient data" << endl;
    return(-1);
  }

  cv::Ptr<cv::face::EigenFaceRecognizer> model = cv::face::EigenFaceRecognizer::create();
  model->train(images, labels);

  // Saved trained model
  std::string model_filename = std::string(argv[2]) + "/eigenfaces_290318.yaml";
  model->save(model_filename);
  cout << "Training successful! Model is saved at: " << model_filename << endl;
  return 0;
}