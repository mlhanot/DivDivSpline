#include <iostream>

#include "../src/divdiv.hpp"

#include "function.hpp"

template<typename F>
double norm(F f) {
  if constexpr (std::is_convertible<F,double>::value) {
    return std::abs(f);
  } else {
    return f.norm();
  }
}
template<typename F>
void print(F f) {
  if constexpr (std::is_convertible<F,double>::value) {
    std::cout<<f;
  } else if constexpr (F::IsVectorAtCompileTime) {
    for (int i = 0; i < f.size(); ++i) {
      std::cout<<((i>0)?", ":"")<<f(i);
    }
  } else {
    for (int i = 0; i < f.rows(); ++i) {
      for (int j = 0; j < f.cols(); ++j) {
        std::cout<<((i>0||j>0)?", ":"")<<f(i,j);
      }
    }
  }
}

template<size_t degree,typename F> requires(degree < 3)
int computeError() {
  constexpr size_t degreep = (degree<2)?degree+1:degree+2;
  std::cout<<"Test derivative using form degree: "<<degree<<std::endl;
  DivDivSpace<degree> V(4);
  DivDivSpace<degreep> Vp(4);
  const Eigen::VectorXd uh = V.interpolate(F::f), duh = Vp.interpolate(F::df);
  std::cout<<"\tComputed interpolate"<<std::endl;
  Eigen::VectorXd dhuh = V.d()*uh;
  if constexpr (degree==2) {
    DivDivSpace<degree+1> Vi(4);
    dhuh = Vi.d()*dhuh;
  }
  const double err = (dhuh-duh).norm();
  std::cout<<"\tNorm of the difference: "<<err<<std::endl;
  return (err > 1e-12);
}

int main() {
  int nbErr = 0;
  nbErr += computeError<0,P0form>();
  nbErr += computeError<1,P1form>();
  nbErr += computeError<2,P2form>();
  if (nbErr > 0) {
    std::cout<<"\033[1;31m"<<nbErr<<" unexpected results\033[0m\n";
  } else {
    std::cout<<nbErr<<" unexpected result\n";
  }
  return nbErr;
}
