#include "../src/divdiv.hpp"

#include <iostream>

#include <slepceps.h>

// Let A = C^T*B_2*C, we have ||Cv||^2 = <Av,v>
// The kernel of A is given by R(D)
// The generalized eigenvectors (v_i,d_i) of A are such that Av_i = l_i Bv_i with l_i =! 0, and Ad_i = 0
// (v_i, d_i) forms a B-orthonormal basis. Since the columns of D span the kernel of A, they are a linear combination of the (d_i). Therefore are B-orthogonal to (v_i) and D^T*B*v_i = 0 \forall i.
// Let At = A + L*B*D*D^T*B
// An eigenvector v_i of A is an eigenvector of At for the same eigenvalue since 
// At*v_i = A*v_i + L*B*D*(D^T*B*v_i) = l_i B*v_i
// The matrix B*D*D^T*B = (BD)*(BD)^T takes value in B*R(D), and since R((BD)^T) = Ker(BD)^⊥, Ker(BD(BD)^T) = Ker((BD)^T) = R(BD)^⊥.
// Therefore B*D*D^T*B is an automorphism of B*R(D).
// Since R(D) is B-orthogonal to (v_i), the l^2 projection of B*R(D) into Span(v_i) is null, and B*R(D) = R(D).
// B^{-1} (B*D*D^T*B) is an automorphism of R(D) and there is a basis of non zero generalized B-eigenvector d_i' of associated eigenvalue l_i' that are B-orthogonal to the (v_i).
// At is inversible, of generalized eigenpair (l_i,v_i) U (L l_i',d_i').
// The (l_i,v_i) are exactly the eigenpair of A, and the d_i' belongs in the kernel of A.
// Since <Atv,v> = ||Cv||^2 + L<D^T*Bv,D^T*Bv> with L >0, all eigenvector are (strictly since At is invertible) positive. 
// The eigenvalues associated with (v_i) are independent of L, but the ones associated with (d_i') are proportional to L.
// Therefore for L large enough, we can ensure that the smallest e.v. of At will be the smallest non-zero e.v. of A
// However, we cannot take L too large without disrupting the solve step.
// Although we cannot predict the correct value of L, we can check a posteriori that the smallest eigenvalue of At is indeed an non-zero eigenvalue of A
// In that case, we are ensured that it is the smallest non-zero e.v. of A
// Any vector v that is B-orthogonal to R(D) can be written as v = a_i v_i
// Therefore:
// ||v||^2 = <a_i v_i,B a_j v_j> = a_ia_j <v_i,Bv_j> = a_i^2 
// ||Cv||^2 = <Av,v> = a_ia_j <Av_i,v_j> = a_ia_j l_i<Bv_i,v_j> = a_i^2 l_i
// l_min ||v||^2 <= ||Cv||^2
// ||v|| <= 1/sqrt(l_min) ||Cv||
// Moreover, Ker(A) = Ker(C). Hence there is c_p = 1/sqrt(l_min) such that for all v B-orthogonal to Ker(C), ||v|| <= c_p ||Cv||. This equality is optimal for v = v_{i_min}. 

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

int main(int argc,char **argv) {
  if (argc < 4) {
    std::cout<<"Usage: formDegree NbCells Lambda\n";
    return 1;
  }
  const int formDegree = std::atoi(argv[1]);
  const int N = std::atoi(argv[2]);
  const double Lambda = std::atof(argv[3]);
  if (formDegree < 0 || formDegree > 2) {
    std::cout<<"formDegree must be between 0 and 2, "<<formDegree<<" was given\n";
    return 1;
  }

  Eigen::SparseMatrix<double> A,At,B,D;
  if (formDegree == 0) { // Treated separatly, the kernel is not given by an operator, but is of small dimension (4)
    DivDivSpace<0> ops1(N);
    DivDivSpace<1> ops2(N);
    Eigen::SparseMatrix<double> C = ops1.d();
    Eigen::SparseMatrix<double> B2 = ops2.L2();
    B = ops1.L2();
    A = C.transpose()*B2*C;
    At = A;
  } else if (formDegree == 1) {
    DivDivSpace<0> ops0(N);
    DivDivSpace<1> ops1(N);
    DivDivSpace<2> ops2(N);
    D = ops0.d(); // DevGrad
    Eigen::SparseMatrix<double> C = ops1.d(); // SymCurl
    Eigen::SparseMatrix<double> B2 = ops2.L2();
    B = ops1.L2();
    A = C.transpose()*B2*C;
    At = A + Lambda*B*D*D.transpose()*B;
  } else { // formDegree == 2
    DivDivSpace<1> ops0(N);
    DivDivSpace<2> ops1(N);
    DivDivSpace<3> ops2i(N);
    DivDivSpace<4> ops2(N);
    D = ops0.d(); // SymCurl
    Eigen::SparseMatrix<double> C = ops2i.d()*ops1.d(); // DivDiv
    Eigen::SparseMatrix<double> B2 = ops2.L2();
    B = ops1.L2();
    A = C.transpose()*B2*C;
    At = A + Lambda*B*D*D.transpose()*B;
  }

  std::cout<<"Matrix A is symm?: "<<(A-Eigen::SparseMatrix<double>(A.transpose())).norm()<<std::endl;
  std::cout<<"Matrix At is symm?: "<<(At-Eigen::SparseMatrix<double>(At.transpose())).norm()<<std::endl;
  std::cout<<"Matrix B is symm?: "<<(B-Eigen::SparseMatrix<double>(B.transpose())).norm()<<std::endl;

  A.makeCompressed();
  At.makeCompressed();
  B.makeCompressed();

  PetscCall(SlepcInitialize(&argc,&argv,NULL,NULL));
  Mat Ats,Bs;
  convertEigenToPETSc(Ats,At);
  convertEigenToPETSc(Bs,B);
  PetscCall(MatSetOption(Bs,MAT_SPD,PETSC_TRUE));
  // Create structures to analyze the results
  Vec vs;
  PetscCall(MatCreateVecs(Ats,&vs,NULL));
  PetscInt nSize;
  PetscCall(VecGetSize(vs,&nSize));
  // Create eigensolver context
  EPS eps;
  PetscCall(EPSCreate(PETSC_COMM_WORLD,&eps));
  PetscCall(EPSSetOperators(eps,Ats,Bs));
  PetscCall(EPSSetProblemType(eps,EPS_GHEP));
  PetscCall(EPSSetTolerances(eps,1e-12,2000));
  if (formDegree == 0) {
    PetscCall(EPSSetDimensions(eps,5,100,PETSC_DEFAULT));
  } else {
    PetscCall(EPSSetDimensions(eps,3,50,PETSC_DEFAULT));
  }
  PetscCall(EPSSetWhichEigenpairs(eps,EPS_SMALLEST_REAL));
  PetscCall(EPSSolve(eps));
  // Print results
  PetscCall(PetscViewerPushFormat(PETSC_VIEWER_STDOUT_WORLD,PETSC_VIEWER_ASCII_INFO_DETAIL));
  PetscCall(EPSConvergedReasonView(eps,PETSC_VIEWER_STDOUT_WORLD));
  PetscCall(EPSErrorView(eps,EPS_ERROR_ABSOLUTE,PETSC_VIEWER_STDOUT_WORLD));
  PetscInt nConv;
  PetscCall(EPSGetConverged(eps,&nConv));
  std::vector<Eigen::VectorXd> v(nConv);
  double eigenVal = -1;
  for (PetscInt i = 0; i < std::min(nConv,7); ++i) {
    PetscScalar egr, egi, *a;
    PetscCall(EPSGetEigenpair(eps,i,&egr,&egi,vs,NULL));
    PetscCall(VecGetArray(vs,&a));
    v[i].resize(nSize);
    for (PetscInt j = 0; j < nSize; ++j) {
      v[i][j] = a[j];
    }
    PetscCall(VecRestoreArray(vs,&a));
    std::cout<<"Eigen value: "<<egr<<", Error (Eigen): "<<(At*v[i] - egr*B*v[i]).norm()<<std::endl;
    std::cout<<"norm: "<<v[i].norm()<<", B-norm: "<<std::sqrt((B*v[i]).dot(v[i]))<<std::endl;
    std::cout<<"A ev: "<<(A*v[i]-egr*B*v[i]).norm()<<std::endl;
    if (formDegree > 0) {
      std::cout<<"DBv: "<<(D.transpose()*B*v[i]).norm()<<std::endl;
    }
    for (int j = 0; j < i; ++j) {
      std::cout<<"B-orthogonality: "<<(B*v[i]).dot(v[j])<<std::endl;
    }
    if (formDegree == 0) {
      if (i < 4) {
        if (egr > 1e-7) { // Check if the first four are indeed in the kernel
          eigenVal = -2.;
        }
      } else if (i == 4) {
        if (eigenVal > -1.5) { // If the first four are in the kernel and the fifth is not, select it
          eigenVal = egr;
        }
      }
    } else { // formDegree > 0
      if (i == 0) {
        if ((A*v[i]-egr*B*v[i]).norm()/v[i].norm() < 1e-7) {
          eigenVal = egr;
        }
      }
    }
  }
  std::cout<<"hmax: "<<1./N<<std::endl;
  if (eigenVal < 0) {
    std::cout<<"Warning: wrong eigenvalue found, try increasing Lambda"<<std::endl;
  } else {
    std::cout<<"For d^"<<formDegree<<" with N = "<<N<<std::endl;
    std::cout<<"Poincare constant: "<<1./std::sqrt(eigenVal)<<std::endl;
  }

  PetscCall(PetscViewerPopFormat(PETSC_VIEWER_STDOUT_WORLD));
  PetscCall(EPSDestroy(&eps));
  PetscCall(VecDestroy(&vs));
  PetscCall(MatDestroy(&Bs));
  PetscCall(MatDestroy(&Ats));
  PetscCall(SlepcFinalize());
  return 0;
}
