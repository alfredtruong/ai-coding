#include "LinearRegression.h"
#include <cmath>
#include <iostream>
#include <vector>

// cost function, least squares
float LinearRegression::OLS_Cost(Eigen::MatrixXd X, Eigen::MatrixXd y, Eigen::VectorXd theta)
{
  // y = mx + b

  // massage into

  // h(x|theta) = theta * x
  //            = theta_0 * x_0 + theta_1 * x_1 where x_0 = 1, theta_0 = b, theta_1 = m

  // loss function
  // J(theta) = 0.5 * sum_{observations x,y} ( h(x|theta) - y ) ^ 2
  // dJ/d(theta_i) = x{i-th observation} - y{i}

  Eigen::MatrixXd inner = pow(((X*theta)-y).array(),2); // squared difference

  return inner.sum() / (2 * X.rows()); // dont agree with " * X.rows() "
}

// compute grad of loss function
std::tuple<Eigen::VectorXd, std::vector<float>> LinearRegression::GradientDescent(Eigen::MatrixXd X, Eigen::MatrixXd y, Eigen::VectorXd theta, float alpha, int iters)
{
  Eigen::MatrixXd temp = theta;
  int parameters = theta.rows();

  std::vector<float> cost;
  cost.push_back(OLS_Cost(X, y, theta));

  for (int i=0; i<iters; i++)
  {
    Eigen::MatrixXd error = X*theta - y;
    for (int j=0; j<parameters; j++)
    {
      Eigen::MatrixXd X_i = X.col(j);
      Eigen::MatrixXd term = error.cwiseProduct(X_i);
      temp(j,0) = theta(j,0) - ((alpha/X.rows())*term.sum());
    }
    theta = temp;
    cost.push_back(OLS_Cost(X, y, theta));
  }
  return std::make_tuple(theta, cost);
}

// compute R2, for evaluating how well we fit the data
float LinearRegression::RSquared(Eigen::MatrixXd y, Eigen::MatrixXd y_hat)
{
  auto num = pow((y-y_hat).array(),2).sum();
  auto den = pow(y.array()-y.mean(),2).sum();

  return 1 - num / den;
}
