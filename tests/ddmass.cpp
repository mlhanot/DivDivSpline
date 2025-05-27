
#include <iostream>

#include "../src/divdiv.hpp"

#include "function.hpp"

template<size_t degree,typename F1, typename F2>
int computeError(double expected) {
  std::cout<<"Test interpolate using form degree: "<<degree<<std::endl;
  DivDivSpace<degree> V(4);
  const Eigen::VectorXd uh1 = V.interpolate(F1::f), uh2 = V.interpolate(F2::f);
  std::cout<<"\tComputed interpolate"<<std::endl;
  const double l2 = uh1.dot(V.L2()*uh2);
  std::cout<<"\tDiscrete L2: "<<l2<<", expected: "<<expected<<std::endl;
  return (std::abs(l2-expected)<1e-10)?0:1;
}

int main() {
  int nbErr = 0;
  constexpr double PE[] = {6514502.0/23625.0,
                           -24743.0/8640.0,
                           27907141.0/126000.0,
                           46813.0/1728.0,
                           281179.0/1350.0,
                           -4657.0/135.0,
                           38339.0/9.0,
                           27268.0/9.0};
  int acc = 0;
  nbErr += computeError<0,P0form,P0form>(PE[acc++]);
  nbErr += computeError<0,P0form,P0formAlt>(PE[acc++]);
  nbErr += computeError<1,P1form,P1form>(PE[acc++]);
  nbErr += computeError<1,P1form,P1formAlt>(PE[acc++]);
  nbErr += computeError<2,P2form,P2form>(PE[acc++]);
  nbErr += computeError<2,P2form,P2formAlt>(PE[acc++]);
  nbErr += computeError<4,P3form,P3form>(PE[acc++]);
  nbErr += computeError<4,P3form,P3formAlt>(PE[acc++]);
  if (nbErr > 0) {
    std::cout<<"\033[1;31m"<<nbErr<<" unexpected results\033[0m\n";
  } else {
    std::cout<<nbErr<<" unexpected result\n";
  }
  return nbErr;
}
