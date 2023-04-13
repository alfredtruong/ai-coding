#include <iostream>
#include "ETL/ETL.h"
#include "LinearRegression/LinearRegression.h"
#include "LogisticRegression/LogisticRegression.h"

/*
https://www.youtube.com/watch?v=jKtbNvCT8Dc // linear regression
https://www.youtube.com/watch?v=4ICxrzWIi3I // logistic regression
*/

using namespace std;

int LinearRegression_main()
{
  // test reading
  ETL etl("wine/wine.data.txt", ",", true);
  std::vector<std::vector<std::string>> dataset = etl.readCSV();
  /*
  for (auto it = dataset.begin(); it < dataset.end(); it++)
  {
    //std::cout << (*it)[0] << std::endl;
    std::cout << it->size() << std::endl;
  }
  */
  int rows = dataset.size();
  int cols = dataset[0].size();

  // test dataframe
  Eigen::MatrixXd dataMat = etl.CSVtoEigen(dataset, rows, cols);
  std::cout << dataMat << std::endl;
  //std::cout << dataMat.rows() << " " << dataMat.cols() << std::endl;

  // test normalization
  Eigen::MatrixXd norm = etl.Normalize(dataMat);
  std::cout << norm << std::endl;
  //std::cout << norm.rows() << " " << norm.cols() << std::endl;

  // test TrainTestSplit
  Eigen::MatrixXd X_train, y_train, X_test, y_test;
  std::tuple<Eigen::MatrixXd, Eigen::MatrixXd, Eigen::MatrixXd, Eigen::MatrixXd> split_data = etl.TrainTestSplit(norm, 0.8);
  std::tie(X_train, y_train, X_test, y_test) = split_data;

  std::cout << X_train.rows() << " " << X_train.cols() << std::endl;
  std::cout << y_train.rows() << " " << y_train.cols() << std::endl;
  std::cout << X_test.rows() << " " << X_test.cols() << std::endl;
  std::cout << y_test.rows() << " " << y_test.cols() << std::endl;


  // test linear regression
  Eigen::VectorXd vec_train = Eigen::VectorXd::Ones(X_train.rows());
  X_train.conservativeResize(X_train.rows(), X_train.cols()+1);
  X_train.col(X_train.cols()-1) = vec_train;

  Eigen::VectorXd vec_test = Eigen::VectorXd::Ones(X_test.rows());
  X_test.conservativeResize(X_test.rows(), X_test.cols()+1);
  X_test.col(X_test.cols()-1) = vec_test;

  Eigen::VectorXd theta = Eigen::VectorXd::Zero(X_train.cols());
  float alpha = 0.01;
  int iters = 1000;

  LinearRegression lr;
  Eigen::VectorXd thetaOut;
  std::vector<float> cost;
  std::tuple<Eigen::VectorXd, std::vector<float>> gd = lr.GradientDescent(X_train, y_train, theta, alpha, iters);
  std::tie(thetaOut, cost) = gd;

  std::cout << "theta = " << thetaOut << std::endl;
  for (auto v : cost)
  {
    std::cout << v << std::endl;
  }

  // export to file
  etl.EigenToFile(thetaOut,"LinearRegression/thetaOut.txt");
  etl.VectorToFile(cost,"LinearRegression/cost.txt");

  // get mean and std
  auto mu_data = etl.Mean(dataMat);
  auto mu_z = mu_data(0,11);

  auto scaled_data = dataMat.rowwise()- dataMat.colwise().mean();
  auto sigma_data = etl.Std(scaled_data);
  auto sigma_z = sigma_data(0,11);

  Eigen::MatrixXd y_train_hat = (X_train * thetaOut * sigma_z).array() + mu_z;
  Eigen::MatrixXd y = dataMat.col(11); //.topRows(20);

  //float R_Squared = lr.RSquared(y,y_train_hat);
  //std::cout << "R-Squared = " << R_Squared << std::endl;
  //etl.EigenToFile(y_train_hat,"LinearRegression/y_train_hat.txt");

  return 0;
}

int LogisticRegression_main()
{
  // test reading
  ETL etl("wine/wine.data.txt", ",", true);
  std::vector<std::vector<std::string>> dataset = etl.readCSV();
  /*
  for (auto it = dataset.begin(); it < dataset.end(); it++)
  {
    //std::cout << (*it)[0] << std::endl;
    std::cout << it->size() << std::endl;
  }
  */
  int rows = dataset.size();
  int cols = dataset[0].size();

  std::cout << rows << std::endl;
  std::cout << cols << std::endl;

  return 0;
}

int main()
{
  LinearRegression_main();
  return 0;
}
