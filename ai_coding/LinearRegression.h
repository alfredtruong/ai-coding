#ifndef __LINEARREGRESSION_H__
#define __LINEARREGRESSION_H__

#include "ETL.h"
#include <Eigen/Dense>

class LinearRegression
{
public:
  LinearRegression() {}

  // fit with gradient descent
  float OLS_Cost(Eigen::MatrixXd X, Eigen::MatrixXd y, Eigen::VectorXd theta);
  std::tuple<Eigen::VectorXd, std::vector<float>> GradientDescent(Eigen::MatrixXd X, Eigen::MatrixXd y, Eigen::VectorXd, float alpha, int iters);

};

#endif // __LINEARREGRESSION_H__
