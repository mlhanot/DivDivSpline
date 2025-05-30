#ifndef SOLVER_HPP
#define SOLVER_HPP

#include <Eigen/Dense>
#include <Eigen/Sparse>

#include <iostream>

class Solver {
  public:
    Solver() {;}
    void setup(const Eigen::SparseMatrix<double> &A) {
      _solver.compute(A);
    }
    Eigen::VectorXd solve(const Eigen::VectorXd &b) {
      return _solver.solve(b);
    }
  private:
    Eigen::SparseLU<Eigen::SparseMatrix<double>> _solver;
};

static int checkSolution(const Eigen::SparseMatrix<double> &A, const Eigen::VectorXd &x, const Eigen::VectorXd &b) {
  const double err = (A*x-b).norm();
  if (err > 1e-7) {
    std::cout<<"Warning: large residual: "<<err<<"\n";
    return 1;
  }
  return 0;
}

#endif
