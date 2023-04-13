#ifndef __ETL_H__
#define __ETL_H__

#include <iostream>
#include <fstream>
#include <Eigen/Dense>
#include <vector>

class ETL
{
  std::string dataset;
  std::string delimiter;
  bool header;

public:
  ETL(std::string data, std::string seperator, bool head) : dataset(data), delimiter(seperator), header(head) {}

  std::vector<std::vector<std::string>> readCSV();
  Eigen::MatrixXd CSVtoEigen(std::vector<std::vector<std::string>> dataset, int rows, int cols);

  Eigen::MatrixXd Normalize(Eigen::MatrixXd data);
  auto Mean(Eigen::MatrixXd data) -> decltype(data.colwise().mean()); // delayed typing
  auto Std(Eigen::MatrixXd data) -> decltype((data.array().square().colwise().sum() / (data.rows()-1)).sqrt()); // delayed typing

  std::tuple<Eigen::MatrixXd, Eigen::MatrixXd, Eigen::MatrixXd, Eigen::MatrixXd> TrainTestSplit(Eigen::MatrixXd data, float train_size);
  void VectorToFile(std::vector<float> vec, std::string filename);
  void EigenToFile(Eigen::MatrixXd mat, std::string filename);

};

#endif // __ETL_H__
