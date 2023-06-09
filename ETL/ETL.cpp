#include "ETL.h"

#include <vector>
#include <stdlib.h>
#include <cmath>
#include <boost/algorithm/string.hpp>

std::vector<std::vector<std::string>> ETL::readCSV() {
  std::ifstream file(dataset);
  std::vector<std::vector<std::string>> dataString;

  std::string line = "";

  while(getline(file,line))
  {
    std::vector<std::string> vec;
    boost::algorithm::split(vec,line,boost::is_any_of(delimiter));
    dataString.push_back(vec);
  }

  file.close();

  return dataString;
}

Eigen::MatrixXd ETL::CSVtoEigen(std::vector<std::vector<std::string>> dataset, int rows, int cols)
{
  if (header == true)
  {
    rows = rows - 1;
  }

  Eigen::MatrixXd mat(cols, rows);
  for (int i=0; i<rows; i++)
  {
    for (int j=0; j<cols; j++)
    {
      mat(j, i) = std::atof(dataset[i][j].c_str());
    }
  }

  return mat.transpose();
}

// column-wise mean
auto ETL::Mean(Eigen::MatrixXd data) -> decltype(data.colwise().mean())
{
  return data.colwise().mean();
}

// column-wise stddev
auto ETL::Std(Eigen::MatrixXd data) -> decltype((data.array().square().colwise().sum() / (data.rows()-1)).sqrt())
{
  return ((data.array().square().colwise().sum()) / (data.rows()-1)).sqrt();
}

Eigen::MatrixXd ETL::Normalize(Eigen::MatrixXd data, bool normalizeTarget)
{
  Eigen::MatrixXd dataNorm;
  if (normalizeTarget == true)
  {
    dataNorm = data;
  }
  else
  {
    dataNorm = data.leftCols(data.cols()-1);
  }

  //std::cout << __func__ << ": " << dataNorm.rows() << std::endl;
  //std::cout << __func__ << ": " << dataNorm.cols() << std::endl;

  auto mean = Mean(dataNorm);
  std::cout << "ok2" << std::endl;
  std::cout << dataNorm.rows() << std::endl;
  std::cout << dataNorm.cols() << std::endl;

  std::cout << mean.rows() << std::endl;
  std::cout << mean.cols() << std::endl;

  Eigen::MatrixXd scaled_data = dataNorm.rowwise() - mean;
  std::cout << "ok3" << std::endl;

  auto stddev = Std(scaled_data);
  Eigen::MatrixXd norm = scaled_data.array().rowwise()/stddev;

  if (normalizeTarget == false)
  {
    norm.conservativeResize(norm.rows(), norm.cols() + 1);
    norm.col(norm.cols()-1) = data.rightCols(1);
  }
  return norm;
}

// make train_test_split
std::tuple<Eigen::MatrixXd, Eigen::MatrixXd, Eigen::MatrixXd, Eigen::MatrixXd> ETL::TrainTestSplit(Eigen::MatrixXd data, float train_size)
{
  int rows = data.rows();
  int train_rows = round(rows * train_size);
  int test_rows = rows - train_rows;

  // training set
  Eigen::MatrixXd train = data.topRows(train_rows);
  Eigen::MatrixXd X_train = train.leftCols(data.cols()-1);
  Eigen::MatrixXd y_train = train.rightCols(1);

  // test set
  Eigen::MatrixXd test = data.bottomRows(test_rows);
  Eigen::MatrixXd X_test = test.leftCols(data.cols()-1);
  Eigen::MatrixXd y_test = test.rightCols(1);

  // return
  return std::make_tuple(X_train, y_train, X_test, y_test);
}

void ETL::VectorToFile(std::vector<float> vec, std::string filename)
{
  std::ofstream output_file(filename);
  std::ostream_iterator<float> output_iterator(output_file, "\n");
  std::copy(vec.begin(), vec.end(), output_iterator);
}

void ETL::EigenToFile(Eigen::MatrixXd mat, std::string filename)
{
  std::ofstream output_file(filename);
  if (output_file.is_open())
  {
    output_file << mat << "\n";
  }
}
