#include <iostream>
#include "ETL.h"

/*
https://www.youtube.com/watch?v=jKtbNvCT8Dc
*/

using namespace std;

int main(int argc, char *argv[])
{
  // test reading
  ETL etl("wine.data.txt", ",", true);

  std::vector<std::vector<std::string>> dataset = etl.readCSV();
  for (auto it = dataset.begin(); it < dataset.end(); it++)
  {
    //std::cout << (*it)[0] << std::endl;
    std::cout << it->size() << std::endl;
  }
  int rows = dataset.size();
  int cols = dataset[0].size();

  // test dataframe
  Eigen::MatrixXd dataMat = etl.CSVtoEigen(dataset, rows, cols);
  std::cout << dataMat << std::endl;

  // test normalization
  Eigen::MatrixXd norm = etl.Normalize(dataMat);
  std::cout << norm << std::endl;

  // test TrainTestSplit
  Eigen::MatrixXd X_train, y_test, X_test, y_test;
  std::tuple<Eigen::MatrixXd, Eigen::MatrixXd, Eigen::MatrixXd, Eigen::MatrixXd> split_data = etl.TrainTestSplit(norm, 0.9);
  return 0;
}
