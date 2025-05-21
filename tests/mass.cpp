
#include <iostream>

#include "../src/tensorspline.hpp"

#include "function.hpp"

template<int _r1,int _r2,int _r3,typename F1, typename F2>
int computeError(double expected) {
  std::cout<<"Test mass using degree: "<<_r1+2<<", "<<_r2+2<<", "<<_r3+2<<" and regularity: "<<_r1<<", "<<_r2<<", "<<_r3<<std::endl;
  TensorSpline<{_r1+2,_r1,_r2+2,_r2,_r3+2,_r3}> ts(4);
  Eigen::VectorXd uh1(ts.nbDofs), uh2(ts.nbDofs);
  size_t acc = 0;
  for (size_t iDim = 0; iDim < 4; ++iDim) {
    for (size_t iT = 0; iT < ts.mesh().nbC[iDim]; ++iT) {
      const size_t localSize = ts.localDim(iDim,iT);
      for (size_t i = 0; i < localSize; ++i) {
        uh1[acc] = ts.interpolate(F1::f,iDim,iT,i,-1);
        uh2[acc] = ts.interpolate(F2::f,iDim,iT,i,-1);
        ++acc;
      }
    }
  }
  std::cout<<"\tComputed interpolate"<<std::endl;
  const double l2 = uh1.dot(ts.L2()*uh2);
  std::cout<<"\tDiscrete L2: "<<l2<<", expected: "<<expected<<std::endl;
  return (std::abs(l2-expected)<1e-10)?0:1;
}

int main() {
  int nbErr = 0;
  constexpr double P11 = 17353./3600., P22 = 64493./20000., P21 = -1363./1800., P33 = 1635695137./92610000., P32 = 518569./72000., P31 = -2204743./960000.;
  nbErr += computeError<1,1,1,P1scalar,P1scalar>(P11);
  nbErr += computeError<0,0,0,P1scalar,P1scalar>(P11);
  nbErr += computeError<0,1,1,P1scalar,P1scalar>(P11);
  nbErr += computeError<-1,-1,-1,P1scalar,P1scalar>(P11);
  nbErr += computeError<1,0,-1,P1scalar,P1scalar>(P11);
  nbErr += computeError<1,1,1,P2scalar,P2scalar>(P22);
  nbErr += computeError<1,1,1,P2scalar,P1scalar>(P21);
  nbErr += computeError<0,0,0,P2scalar,P2scalar>(P22);
  nbErr += computeError<0,0,0,P2scalar,P1scalar>(P21);
  nbErr += computeError<0,1,1,P2scalar,P2scalar>(P22);
  nbErr += computeError<0,1,1,P2scalar,P1scalar>(P21);
  nbErr += computeError<1,1,1,P3scalar,P3scalar>(P33);
  nbErr += computeError<1,1,1,P3scalar,P2scalar>(P32);
  nbErr += computeError<1,1,1,P3scalar,P1scalar>(P31);
  if (nbErr > 0) {
    std::cout<<"\033[1;31m"<<nbErr<<" unexpected results\033[0m\n";
  } else {
    std::cout<<nbErr<<" unexpected result\n";
  }
  return nbErr;
}
