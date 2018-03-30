// Reference: https://en.wikipedia.org/wiki/Eigenface

#include <iostream>
#include <string>
#include <vector>

#include <opencv2/opencv.hpp>

#define HEIGHT 100
#define WIDTH  100

using namespace std;

static int read_database(const std::string& file_name, std::vector<cv::Mat>& images, std::vector<int>& labels, char sep = ',')
{
  cout << "Reading images from " << file_name << "... " << flush;
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
      cv::Mat tmp = cv::imread(path_str, CV_LOAD_IMAGE_GRAYSCALE);        // [100 x 100]
      if(tmp.cols != WIDTH || tmp.rows != HEIGHT)
      {
        cout << "WARN: Image of invalid dimension " << tmp.size() << " at " << path_str << endl;
        cv::resize(tmp, tmp, cv::Size(HEIGHT, WIDTH));
      }
      images.push_back(tmp);
      labels.push_back(std::stoi(label_str));
    }
    else
    {
      cout << "Invalid data at: " << path_str << endl;
    }
  }

  cout << "Done. Read " << images.size() << " images." << endl;
  return 1;
}

static void createDataMatrix(const std::vector<cv::Mat> images, cv::Mat &m_data, cv::Mat &m_data_mean)
{
  cout << "Flattening, zero-meaning, and storing all images into a matrix... " << flush;

  m_data.create(0, 0, CV_64FC1);                                          // [N x 10000]
  cv::Mat data_sum(cv::Size(HEIGHT * WIDTH, 1), CV_64FC1, cv::Scalar(0)); // [1 x 10000]
  for(int i = 0, i_end = images.size(); i < i_end; i++)
  {
    // Flatten and append to data matrix
    cv::Mat tmp = images[i].reshape(1,1);                                 // [1 x 10000]
    tmp.convertTo(tmp, CV_64FC1);
    m_data.push_back(tmp);

    cv::add(data_sum, tmp, data_sum);
  }

  // Zero-mean
  m_data_mean.create(1, 10000, CV_64FC1);                                 // [1 x 10000]
  cv::multiply(data_sum, 1.0 / images.size(), m_data_mean);
  cv::Mat data_mean_spreaded = cv::repeat(m_data_mean, m_data.rows, 1);   // [N x 10000], used for quick subtraction
  cv::subtract(m_data, data_mean_spreaded, m_data);

  cv::transpose(m_data, m_data);                                          // [10000 x N]
  cv::transpose(m_data_mean, m_data_mean);                                // [10000 x 1]
  cout << "Done. Data size is " << m_data.size << endl;
}

static cv::Mat computeCovarMatrix(const cv::Mat input)
{
  cout << "Computing covariance matrix... " << flush;
  // devide by n if sample size > 25, by n - 1 if sample size < 25
  // C = B* . B / (n - 1)
  cv::Mat covar_mat(0, 0, CV_32FC1);                                      // [10000 x 10000] !!
  cv::mulTransposed(input, covar_mat, false); // means output = input x input.T
  cv::multiply(covar_mat, cv::Scalar(1.0 / input.rows), covar_mat);

  cout << "Done. Covariance matrix is " << covar_mat.size << endl;
  return covar_mat;
}

int main(int argc, char** argv)
{
  std::string fconfig_name;
  if(argc < 2)
  {
    cout << "WARN: No configuration file is indicated. Using ../default_config.yaml" << endl;
    cout << "Usage: ./trainEigenfaces \"config_file.yaml\"" << endl;
    fconfig_name = "../default_config.yaml";
  }
  else fconfig_name = argv[1];

  cv::FileStorage config(fconfig_name, cv::FileStorage::READ);

  std::vector<cv::Mat> images;
  std::vector<int> labels;
  std::string image_index_file;
  config["Image.IndexFile"] >> image_index_file;

  if(!read_database(image_index_file, images, labels)) 
    return(-1);
  if(images.size() < 2)
  {
    cout << "ERROR: Insufficient data (" << images.size() << ")." << endl;
    return(-1);
  }

  // Flatten and append all images into a matrix
  // This operation flattens and extract the average image from every image
  // Each image is stored as a col
  cv::Mat m_data;                                         // [10000 x N]
  cv::Mat m_data_mean;                                    // [10000 x 1]
  createDataMatrix(images, m_data, m_data_mean);

  std::string compute_eigen;
  config["Eigen.Recompute"] >> compute_eigen;

  cv::Mat m_eigenvalues;                                  // [200 x 1]
  cv::Mat m_eigenvectors;                                 // [200 x 10000] note that each eigenvector is stored as row
  if(compute_eigen == "true")
  {
    // Compute matrix covariance
    cv::Mat m_cov = computeCovarMatrix(m_data);           // [10000 x 10000] !!

    // Calculate Eigenvalues and Eigenvectors of covariance matrix
    // Eigenvalues are stored in descending order
    // Eigenvectors are stored in rows, in the corresponding order and normalized
    cout << "Computing Eigenvalues and Eigenvectors (may take a very long time!)... " << flush;
    cv::Mat eigen_values;                                 // [10000 x 1]
    cv::Mat eigen_vectors;                                // [10000 x 10000] note that each eigenvector is stored as row
    if(!cv::eigen(m_cov, eigen_values, eigen_vectors))
    {
      cout << "Failed to calculate Eigenvalues and Eigenvectors. Abort." << endl;
      return(-1);
    }
    else cout << "Done." << endl;

    // Evaluate the required number of eigenfaces for representation
    const double required_covar = cv::sum(eigen_values)[0] * 0.95;
    double total_covar = 0;
    int eigenfaces_size = WIDTH * HEIGHT;
    for(int i = 0, i_end = WIDTH * HEIGHT; i < i_end; i++)
    {
      total_covar += eigen_values.at<double>(0, i);
      if(total_covar >= required_covar)
      {
        eigenfaces_size = i + 1;
        break;
      }
    }
    cout << eigenfaces_size << " Eigenfaces are required to cover 95%% covariance." << endl;
    int eigenfaces_required_size = (int)config["Eigen.RequiredSize"];
    eigenfaces_size = eigenfaces_size >= eigenfaces_required_size ? eigenfaces_size : eigenfaces_required_size;
    cout << "Taking the first " << eigenfaces_size << " eigenfaces for the model." << endl;

    // Saved full eigen
    std::string output_dir;
    config["Output.Directory"] >> output_dir;

    std::string out_full_filename = (string)config["Output.FullModel"];
    cv::FileStorage fsf(out_full_filename, cv::FileStorage::WRITE);
    fsf << "mean" << m_data_mean;
    fsf << "eigenvalues" << eigen_values;
    fsf << "eigenvectors" << eigen_vectors;
    fsf.release();
    cout << "Full model is saved at: " << out_full_filename << endl;

    // Saved reduced eigen
    std::string out_filename = (string)config["Output.ReducedModel"];
    cv::FileStorage fsr(out_filename, cv::FileStorage::WRITE);
    fsr << "mean" << m_data_mean;
    fsr << "eigenvalues" << eigen_values(cv::Range(0, eigenfaces_size), cv::Range::all());
    fsr << "eigenvectors" << eigen_vectors(cv::Range(0, eigenfaces_size), cv::Range::all());
    fsr.release();
    cout << "Reduced model is saved at: " << out_filename << endl;

    m_eigenvalues = eigen_values(cv::Range(0, eigenfaces_size), cv::Range::all());    // [200 x 1] from [10000 x 1]
    eigen_values.deallocate();
    m_eigenvectors = eigen_vectors(cv::Range(0, eigenfaces_size), cv::Range::all());  // [200 x 10000] from [10000 x 10000]
    eigen_vectors.deallocate();
  }
  else if(compute_eigen == "false")
  {
    std::string in_filename = (string)config["Eigen.ReducedModel"];
    cv::FileStorage fsi(in_filename, cv::FileStorage::READ);

    fsi["eigenvalues"] >> m_eigenvalues;                  // [200 x 1]
    cout << "Read Eigenvalues of size " << m_eigenvalues.size << endl;

    fsi["eigenvectors"] >> m_eigenvectors;                // [200 x 10000]
    cout << "Read Eigenvectors of size " << m_eigenvectors.size << endl;

    // Sanity check
    if(m_eigenvalues.rows != m_eigenvectors.rows)
    {
      cout << "ERROR: Conflicting sizes of eigenvalues and eigenvectors." << endl;
      return(-1);
    }

    cv::Mat saved_mean;
    fsi["mean"] >> saved_mean;
    if(saved_mean.size != m_data_mean.size)
    {
      cout << "ERROR: Data mean size changed! (" << saved_mean.size << " and " << m_data_mean.size << ")!" << endl;
      return(-1);
    }
  }
  else 
  {
    cout << "ERROR: Invalid Eigen.Recompute value in config file (" << compute_eigen << ")" << endl;
    return(-1);
  }

  // Calculate face representation in the new PCA space: Eigenfaces
  // Each Eigenface is stored as a col
  cv::Mat m_eigenfaces = m_eigenvectors * m_data;         // [200 x N]

  // Store model
  std::string model_filename = (string)config["Output.EigenfacesModel"];
  cv::FileStorage fsmodel(model_filename, cv::FileStorage::WRITE);
  // also write label for recognition use
  cv::Mat m_labels(labels);                               // [N x 1]
  fsmodel << "labels" << m_labels;
  fsmodel << "mean" << m_data_mean;
  fsmodel << "eigenvalues" << m_eigenvalues;
  fsmodel << "eigenvectors" << m_eigenvectors;
  fsmodel << "eigenfaces" << m_eigenfaces;
  fsmodel.release();
  cout << "Successfully saving Eigenfaces model at: " << model_filename << endl;

  return 0;
}