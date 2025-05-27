#include <iostream>

#include "../src/divdiv.hpp"

#include <Eigen/QR>

double lInftySparse(const Eigen::SparseMatrix<double> &M) {
  Eigen::Index nnz = M.nonZeros();
  return Eigen::VectorXd::Map(M.valuePtr(),nnz).lpNorm<Eigen::Infinity>();
}

template<size_t degree> requires(degree < 2)
int testComplex() {
  std::cout<<"Test complex using form degree: "<<degree<<std::endl;
  DivDivSpace<degree> V(4);
  DivDivSpace<degree+1> Vp(4);
  Eigen::SparseMatrix<double> dd = Vp.d()*V.d();
  if constexpr (degree==1) {
    DivDivSpace<degree+2> Vi(4);
    dd = Vi.d()*dd;
  }
  const double err = lInftySparse(dd);
  std::cout<<"\tL^infty norm of d^"<<degree+1<<"d^"<<degree<<": "<<err<<std::endl;
  return (err > 1e-12);
}

int testCohomology() {
  DivDivSpace<0> V0(4);
  DivDivSpace<1> V1(4);
  DivDivSpace<2> V2(4);
  DivDivSpace<3> V3(4);
  DivDivSpace<4> V4(4);
  Eigen::MatrixXd d0 = V0.d(), d1 = V1.d(), d2 = V3.d()*V2.d();

  const int rank0 = d0.colPivHouseholderQr().rank();
  std::cout<<"Rank 0 computed"<<std::flush;
  const int rank1 = d1.colPivHouseholderQr().rank();
  std::cout<<"\rRank 1 computed"<<std::flush;
  const int rank2 = d2.colPivHouseholderQr().rank();
  std::cout<<"\rRank 2 computed"<<std::flush;
  const int dim0 = d0.cols(), dim1 = d1.cols(), dim2 = d2.cols();
  const int ker3 = V4.nbDofs;

  std::cout<<"\rKer(d0): "<<dim0-rank0<<", Rank(d0): "<<rank0
           <<", Ker(d1): "<<dim1-rank1<<", Rank(d1): "<<rank1
           <<", Ker(d2): "<<dim2-rank2<<", Rank(d2): "<<rank2
           <<", Ker(d3): "<<ker3<<std::endl;
  int err = 0;
  if (dim0-rank0 != 4) ++err;
  if (rank0 != dim1-rank1) ++err;
  if (rank1 != dim2-rank2) ++err;
  if (rank2 != ker3) ++err;
  return err;
}

int main() {
  int nbErr = 0;
  nbErr += testComplex<0>();
  nbErr += testComplex<1>();
  nbErr += testCohomology();
  if (nbErr > 0) {
    std::cout<<"\033[1;31m"<<nbErr<<" unexpected results\033[0m\n";
  } else {
    std::cout<<nbErr<<" unexpected result\n";
  }
  return nbErr;
}
