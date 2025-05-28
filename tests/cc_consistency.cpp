#include "../src/divdiv.hpp"
#include "function_adj.hpp"

#include <iostream>

#include <petscksp.h>

// |||L||| := sup |L(x)|/||x||
// |||L||| = sup_{L(x) = 1} 1/||x|| = 1/(inf_{L(x) = 1} ||x||)
// ||x||^2 := x^T M x 
// |||L||| = 1/||x_min||, where x_min satisfy min 1/2 x^T 2M x, s.t. Lx = 1
// This is a quadratic problem with linear equality, the solution x satisfies 
// | 2M L^T | |x| = |0|
// | L   0  | |l|   |1| 
// Using Schur complement we can see that x_min = (2M)^{-1}L^T / (L (2M)^{-1}L^T) = M^{-1}L^T/(LM^{-1}L^T)
// Therefore, ||x_min||^2 = 1/(LM^{-1}L^T)^2 L M^{-T} M M^{-1} L^T = L M^{-1} L^T/(LM^{-1}L^T)^2 = 1/(LM^{-1}L^T), since M^{-T} = M^{-1}
// This leads to |||L||| = \sqrt{LM^{-1}L^T} 

void computeOperatorNorm(const Eigen::SparseMatrix<double> &M, const Eigen::VectorXd &L) {
  Eigen::SparseLU<Eigen::SparseMatrix<double>> solver;
  solver.compute(M);
  Eigen::VectorXd MinvL = solver.solve(L);
  const double opsNorm = std::sqrt(MinvL.dot(L));
  std::cout<<"normL: "<<L.norm()<<", solver: "<<(M*MinvL - L).norm()<<std::endl;
  std::cout<<"Operator norm: "<<opsNorm<<std::endl;
}

PetscErrorCode convertEigenToPETSc (Mat &Apetsc, const Eigen::SparseMatrix<double,Eigen::ColMajor> &A) {
  PetscInt m = A.rows();
  PetscInt n = A.cols();
  int nnz = A.nonZeros();
  std::cout<<"m: "<<m<<", n:"<<n<<", nnz: "<<nnz<<", filled: "<<double(nnz)/(double(n)*double(m))<<std::endl;
  std::vector<PetscInt> nnzCols(n);
  for (PetscInt Icol = 0; Icol < n; ++Icol) {
    nnzCols[Icol] = A.outerIndexPtr()[Icol+1] - A.outerIndexPtr()[Icol];
  } // We assume the matrix symmetric, hence the number of non zero par column is the same as the number per row
  PetscCall(MatCreateSeqAIJ(PETSC_COMM_WORLD,m,n,0,nnzCols.data(),&Apetsc));
  PetscCall(MatSetOption(Apetsc,MAT_SYMMETRIC,PETSC_TRUE));
  PetscCall(MatSetOption(Apetsc,MAT_ROW_ORIENTED,PETSC_FALSE));
  for (PetscInt Icol = 0; Icol < n; ++Icol) {
    int offset = A.outerIndexPtr()[Icol];
    PetscCall(MatSetValues(Apetsc,nnzCols[Icol],A.innerIndexPtr()+offset,1,&Icol,A.valuePtr()+offset,INSERT_VALUES));
  }
  PetscCall(MatAssemblyBegin(Apetsc,MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(Apetsc,MAT_FINAL_ASSEMBLY));
  return 0;
}
PetscErrorCode convertEigenToPETSc (Vec &Vpetsc, const Eigen::VectorXd &V) {
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
PetscErrorCode computeOperatorNormPETSc(const Eigen::SparseMatrix<double> &M, const Eigen::VectorXd &L) {
  Mat A;
  convertEigenToPETSc(A,M);
  Vec b, x;
  PetscCall(MatCreateVecs(A,&b,NULL));
  PetscCall(MatCreateVecs(A,&x,NULL));
  convertEigenToPETSc(b,L);
  // Create solver context
  KSP ksp;
  PetscCall(KSPCreate(PETSC_COMM_WORLD,&ksp));
  PetscCall(KSPSetOperators(ksp,A,A));
  PetscCall(KSPSetFromOptions(ksp));
  PetscCall(KSPSolve(ksp,b,x));
  Eigen::VectorXd MinvL(L.size());
  PetscScalar *a;
  PetscCall(VecGetArray(x,&a));
  for (PetscInt j = 0; j < L.size(); ++j) {
    MinvL[j] = a[j];
  }
  PetscCall(VecRestoreArray(x,&a));
  PetscCall(KSPDestroy(&ksp));
  PetscCall(VecDestroy(&x));
  PetscCall(VecDestroy(&b));
  PetscCall(MatDestroy(&A));
  const double opsNorm = std::sqrt(MinvL.dot(L));
  std::cout<<"normL: "<<L.norm()<<", solver: "<<(M*MinvL - L).norm()<<std::endl;
  std::cout<<"Operator norm: "<<opsNorm<<std::endl;
  return 0;
}

int main(int argc,char **argv) {
  if (argc < 5) {
    std::cout<<"Usage: formDegree NbCells SolverType[Eigen only,auto, petsconly] Case\n";
    return 1;
  }
  const int formDegree = std::atoi(argv[1]);
  const int N = std::atoi(argv[2]);
  const int solverType = std::atoi(argv[3]);
  const int caseU = std::atoi(argv[4]);
  if (formDegree < 0 || formDegree > 2) {
    std::cout<<"formDegree must be between 0 and 2, "<<formDegree<<" was given\n";
    return 1;
  }

  if (solverType > 0) {
    PetscCall(PetscInitialize(&argc,&argv,NULL,NULL));
  }

  Eigen::SparseMatrix<double> Omega;
  Eigen::VectorXd L;
  auto setMat = [&Omega,&L,N]<size_t k,typename F>() {
    DivDivSpace<k> ops1(N);
    DivDivSpace<(k==2)?4:k+1> ops2(N);
    Eigen::SparseMatrix<double> M1, M2, dh, BC;
    M1 = ops1.L2();
    M2 = ops2.L2();
    dh = ops1.d();
    if constexpr (k == 2) {
      DivDivSpace<3> opsi(N);
      dh = opsi.d()*dh;
    }
    ops1.markBoundary(ops1.allBoundary);
    BC = ops1.interiorExtension();
    Omega = BC.transpose()*(M1 + dh.transpose()*M2*dh)*BC;
    std::cout<<"Mass matrix assembled"<<std::endl;
    Eigen::VectorXd Iu, IdeltaU;
    Iu = ops2.interpolate(F::f);
    std::cout<<"Interpolate of U computed"<<std::endl;
    IdeltaU = ops1.interpolate(F::deltaf);
    std::cout<<"Interpolate of deltaU computed"<<std::endl;
    L = (Iu.transpose()*M2*dh - IdeltaU.transpose()*M1)*BC;
  };
  if (formDegree == 0) { 
    if (caseU == 0) {
      setMat.template operator()<0,P1C0>();
    } else if (caseU == 1) {
      setMat.template operator()<0,P1C1>();
    } else {
      setMat.template operator()<0,P1C2>();
    }
  } else if (formDegree == 1) {
    if (caseU == 0) {
      setMat.template operator()<1,P2C0>();
    } else if (caseU == 1) {
      setMat.template operator()<1,P2C1>();
    } else {
      setMat.template operator()<1,P2C2>();
    }
  } else {
    if (caseU == 0) {
      setMat.template operator()<2,P3C0>();
    } else if (caseU == 1) {
      setMat.template operator()<2,P3C1>();
    } else {
      setMat.template operator()<2,P3C2>();
    }
  }
  std::cout<<"For d^"<<formDegree<<" with N = "<<N<<", testCase: "<<caseU<<std::endl;
  if (solverType == 0) {
    computeOperatorNorm(Omega,L);
  } else if (solverType == 1) {
    if (L.size() < 2e4) {
      computeOperatorNorm(Omega,L);
    } else {
      computeOperatorNormPETSc(Omega,L);
    }
  } else {
    computeOperatorNormPETSc(Omega,L);
  }

  if (solverType > 0) {
    PetscCall(PetscFinalize());
  }
  return 0;
}
