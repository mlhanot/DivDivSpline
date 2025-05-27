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

template<size_t degree,typename F> requires(degree < 5)
int computeError() {
  std::cout<<"Test interpolate using form degree: "<<degree<<std::endl;
  DivDivSpace<degree> V(4);
  using Vtype = std::tuple_element<degree,ValueTypes>::type;
  const Eigen::VectorXd uh = V.interpolate(F::f);
  std::cout<<"\tComputed interpolate"<<std::endl;
  auto evalLoc = [&V,&uh](const Eigen::Vector3d &x)->int{
    const Vtype valI = V.evaluate(x,uh), valF = mergeComp<degree>(F::f,x,0,0,0);
    std::cout<<"\tValue at: "<<x.transpose()<<", Interpolate: ";
    print(valI);
    std::cout<<", Function: ";
    print(valF);
    std::cout<<std::endl;
    return norm(valI-valF) > 1e-12;
  };
  int nbErr = 0;
  nbErr += evalLoc({0,0,0});
  nbErr += evalLoc({0.5,0,0});
  nbErr += evalLoc({0.2,0.11,0.57});
  return nbErr;
}

int main() {
  int nbErr = 0;
  nbErr += computeError<0,P0form>();
  nbErr += computeError<1,P1form>();
  nbErr += computeError<2,P2form>();
  nbErr += computeError<4,P3form>();
  if (nbErr > 0) {
    std::cout<<"\033[1;31m"<<nbErr<<" unexpected results\033[0m\n";
  } else {
    std::cout<<nbErr<<" unexpected result\n";
  }
  return nbErr;
}
