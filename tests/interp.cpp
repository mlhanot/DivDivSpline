
#include <iostream>

#include "../src/tensorspline.hpp"

#include "function.hpp"

template<int _r1,int _r2,int _r3,typename F>
int computeError() {
  std::cout<<"Test interpolate using degree: "<<_r1+2<<", "<<_r2+2<<", "<<_r3+2<<" and regularity: "<<_r1<<", "<<_r2<<", "<<_r3<<std::endl;
  TensorSpline<{_r1+2,_r1,_r2+2,_r2,_r3+2,_r3}> ts(4);
  Eigen::VectorXd uh(ts.nbDofs);
  size_t acc = 0;
  for (size_t iDim = 0; iDim < 4; ++iDim) {
    for (size_t iT = 0; iT < ts.mesh().nbC[iDim]; ++iT) {
      const size_t localSize = ts.localDim(iDim,iT);
      for (size_t i = 0; i < localSize; ++i) {
        uh[acc++] = ts.interpolate(F::f,iDim,iT,i,-1);
      }
    }
  }
  std::cout<<"\tComputed interpolate"<<std::endl;
  auto evalLoc = [&ts,&uh](const Eigen::Vector3d &x)->int{
    const double valI = ts.evaluate(x,uh), valF = F::f(x,0,0,0);
    std::cout<<"\tValue at: "<<x.transpose()<<", Interpolate: "<<valI<<", Function: "<<valF<<std::endl;
    return std::abs(valI-valF) > 1e-12;
  };
  int nbErr = 0;
  nbErr += evalLoc({0,0,0});
  nbErr += evalLoc({0.5,0,0});
  nbErr += evalLoc({0.2,0.11,0.57});
  return nbErr;
}

int main() {
  int nbErr = 0;
  nbErr += computeError<1,1,1,P1scalar>();
  nbErr += computeError<0,0,0,P1scalar>();
  nbErr += computeError<0,1,1,P1scalar>();
  nbErr += computeError<-1,-1,-1,P1scalar>();
  nbErr += computeError<1,0,-1,P1scalar>();
  nbErr += computeError<1,1,1,P2scalar>();
  nbErr += computeError<0,0,0,P2scalar>();
  nbErr += computeError<0,1,1,P2scalar>();
  nbErr += computeError<1,1,1,P3scalar>();
  if (nbErr > 0) {
    std::cout<<"\033[1;31m"<<nbErr<<" unexpected results\033[0m\n";
  } else {
    std::cout<<nbErr<<" unexpected result\n";
  }
  return nbErr;
}
