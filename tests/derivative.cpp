
#include <iostream>

#include "../src/tensorspline.hpp"

#include "function.hpp"

template<int _r1,int _r2,int _r3,typename F>
int computeError() {
  std::cout<<"Test derivative using degree: "<<_r1+2<<", "<<_r2+2<<", "<<_r3+2<<" and regularity: "<<_r1<<", "<<_r2<<", "<<_r3<<std::endl;
  TensorSpline<{_r1+2,_r1,_r2+2,_r2,_r3+2,_r3}> ts(4);
  Eigen::VectorXd uh(ts.nbDofs);
  size_t acc = 0;
  for (size_t iDim = 0; iDim < 4; ++iDim) {
    for (size_t iT = 0; iT < ts.mesh().nbC[iDim]; ++iT) {
      const size_t localSize = ts.localDim(iDim,iT);
      for (size_t i = 0; i < localSize; ++i) {
        uh[acc++] = ts.interpolate(F::f,iDim,iT,i,3);
      }
    }
  }
  std::cout<<"\tComputed interpolate"<<std::endl;
  auto evalLocD = [&ts,&uh]<unsigned direction>(const Eigen::Vector3d &x)->int{
    TensorSpline<ts.template derivativeSpace<direction>()> dts(4);
    Eigen::SparseMatrix<double> dx = ts.template derivative<direction>();
    Eigen::VectorXd dxuh = dx*uh;
    const double valI = dts.evaluate(x,dxuh), valF = F::f(x,(direction==0)?1:0,(direction==1)?1:0,(direction==2)?1:0);
    std::cout<<"\tValue at: "<<x.transpose()<<", direction : "<<direction<<", Interpolate: "<<valI<<", Function: "<<valF<<std::endl;
    return std::abs(valI-valF) > 1e-12;
  };
  auto evalLoc = [&evalLocD](const Eigen::Vector3d &x)->int{
    int rv = 0;
    if constexpr (_r1 >= 0) {
      rv += evalLocD.template operator()<0>(x);
    } else {
      std::cout<<"\tSkipping derivative along x since the space lacks regularity\n";
    }
    if constexpr (_r2 >= 0) {
      rv += evalLocD.template operator()<1>(x);
    } else {
      std::cout<<"\tSkipping derivative along y since the space lacks regularity\n";
    }
    if constexpr (_r3 >= 0) {
      rv += evalLocD.template operator()<2>(x);
    } else {
      std::cout<<"\tSkipping derivative along z since the space lacks regularity\n";
    }
    return rv;
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
