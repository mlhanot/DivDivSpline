#ifndef SOLVER_HPP
#define SOLVER_HPP

#include <petscksp.h>
#include <Eigen/Dense>
#include <Eigen/Sparse>

#include <iostream>

class Solver {
  public:
    virtual ~Solver() {;}
    void virtual setup(const Eigen::SparseMatrix<double> &A) = 0;
    Eigen::VectorXd virtual solve(const Eigen::VectorXd &b) = 0;
};

class SolverLU final : public Solver {
  public:
    void setup(const Eigen::SparseMatrix<double> &A) override {
      _solver.compute(A);
    }
    Eigen::VectorXd solve(const Eigen::VectorXd &b) override {
      return _solver.solve(b);
    }
  private:
    Eigen::SparseLU<Eigen::SparseMatrix<double>> _solver;
};

class SolverPetsc final : public Solver {
  public:
    void setup(const Eigen::SparseMatrix<double> &A) override {
      convertEigenToPETSc(_A,A);
        PetscCallAbort(PETSC_COMM_WORLD,MatCreateVecs(_A,&_b,NULL));
        PetscCallAbort(PETSC_COMM_WORLD,MatCreateVecs(_A,&_x,NULL));
        PetscCallAbort(PETSC_COMM_WORLD,KSPCreate(PETSC_COMM_WORLD,&_ksp));
        PetscCallAbort(PETSC_COMM_WORLD,KSPSetOperators(_ksp,_A,_A));
        PetscCallAbort(PETSC_COMM_WORLD,KSPSetFromOptions(_ksp));
        PetscCallAbort(PETSC_COMM_WORLD,KSPSetInitialGuessNonzero(_ksp,PETSC_TRUE));
      Eigen::VectorXd z = Eigen::VectorXd::Zero(A.cols());
      convertEigenToPETSc(_x,z);
    }
    Eigen::VectorXd solve(const Eigen::VectorXd &b) override {
      convertEigenToPETSc(_b,b);
      PetscCallAbort(PETSC_COMM_WORLD,KSPSolve(_ksp,_b,_x));
      Eigen::VectorXd x(b.size());
      PetscScalar *a;
      VecGetArray(_x,&a);
      for (PetscInt j = 0; j < b.size(); ++j) {
        x[j] = a[j];
      }
      VecRestoreArray(_x,&a);
      return x;
    }
    ~SolverPetsc() {
      KSPDestroy(&_ksp);
      VecDestroy(&_x);
      VecDestroy(&_b);
      MatDestroy(&_A);
    }
  private:
    PetscErrorCode static convertEigenToPETSc (Mat &Apetsc, const Eigen::SparseMatrix<double,Eigen::ColMajor> &A) {
      PetscInt m = A.rows();
      PetscInt n = A.cols();
      int nnz = A.nonZeros();
      std::cout<<"m: "<<m<<", n:"<<n<<", nnz: "<<nnz<<", filled: "<<double(nnz)/(double(n)*double(m))<<std::endl;
      std::vector<PetscInt> nnzCols(n);
      for (PetscInt Icol = 0; Icol < n; ++Icol) {
        nnzCols[Icol] = A.outerIndexPtr()[Icol+1] - A.outerIndexPtr()[Icol];
      } // We assume the matrix to have a symmetric patern, hence the number of non zero par column is the same as the number per row
      PetscCall(MatCreateSeqAIJ(PETSC_COMM_WORLD,m,n,0,nnzCols.data(),&Apetsc));
      PetscCall(MatSetOption(Apetsc,MAT_ROW_ORIENTED,PETSC_FALSE));
      for (PetscInt Icol = 0; Icol < n; ++Icol) {
        int offset = A.outerIndexPtr()[Icol];
        PetscCall(MatSetValues(Apetsc,nnzCols[Icol],A.innerIndexPtr()+offset,1,&Icol,A.valuePtr()+offset,INSERT_VALUES));
      }
      PetscCall(MatAssemblyBegin(Apetsc,MAT_FINAL_ASSEMBLY));
      PetscCall(MatAssemblyEnd(Apetsc,MAT_FINAL_ASSEMBLY));
      return 0;
    }
    PetscErrorCode static convertEigenToPETSc (Vec &Vpetsc, const Eigen::VectorXd &V) {
      PetscInt n = V.size();
      std::vector<PetscInt> ix(n);
      for (PetscInt Icol = 0; Icol < n; ++Icol) {
        ix[Icol] = Icol;
      } 
      PetscCall(VecSetValues(Vpetsc, n, ix.data(), V.data(),INSERT_VALUES));
      PetscCall(VecAssemblyBegin(Vpetsc));
      PetscCall(VecAssemblyEnd(Vpetsc));
      return 0;
    }
    Mat _A;
    Vec _b, _x;
    KSP _ksp;
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
