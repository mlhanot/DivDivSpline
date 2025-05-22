#ifndef FUNCTION_HPP
#define FUNCTION_HPP

#include <Eigen/Dense>
#include "autodiff/forward/real.hpp"
#include "autodiff/forward/dual.hpp"

/// Wrapper for first order derivative
template<typename F>
double autodiff1st(F ADf, const Eigen::Vector3d &X, unsigned dx, unsigned dy, unsigned dz) {
  assert(dx + dy + dz < 2 && "only first order derivative implemented");
  autodiff::real x = X(0), y = X(1), z = X(2);
  if (dx > 0) {
    return autodiff::derivative(ADf,autodiff::wrt(x), autodiff::at(x,y,z));
  } else if (dy > 0) {
    return autodiff::derivative(ADf,autodiff::wrt(y), autodiff::at(x,y,z));
  } else if (dz > 0) {
    return autodiff::derivative(ADf,autodiff::wrt(z), autodiff::at(x,y,z));
  } else {
    return ADf(x,y,z).val();
  }
};

/// Wrapper for thrid order derivative
template<typename F>
double autodiff3rd(F ADf, const Eigen::Vector3d &X, unsigned dx, unsigned dy, unsigned dz) {
  assert(dx < 2 && dy < 2 && dz < 2 && "only first order derivative implemented");
  autodiff::dual3rd x = X(0), y = X(1), z = X(2);
  if (dx > 0) {
    if (dy > 0) {
      if (dz > 0) {
        return autodiff::derivatives(ADf,autodiff::wrt(x,y,z), autodiff::at(x,y,z))[3];
      } else {
        return autodiff::derivatives(ADf,autodiff::wrt(x,y), autodiff::at(x,y,z))[2];
      }
    } else {
      if (dz > 0) {
        return autodiff::derivatives(ADf,autodiff::wrt(x,z), autodiff::at(x,y,z))[2];
      } else {
        return autodiff::derivatives(ADf,autodiff::wrt(x), autodiff::at(x,y,z))[1];
      }
    }
  } else if (dy > 0) { // dx == 0
    if (dz > 0) {
      return autodiff::derivatives(ADf,autodiff::wrt(y,z), autodiff::at(x,y,z))[2];
    } else {
      return autodiff::derivatives(ADf,autodiff::wrt(y), autodiff::at(x,y,z))[1];
    }
  } else if (dz > 0) { // dx == 0 && dy == 0
    return autodiff::derivatives(ADf,autodiff::wrt(z), autodiff::at(x,y,z))[1];
  } else {
    return autodiff::val(ADf(x,y,z));
  }
};

struct P1scalar {
  static double f(const Eigen::Vector3d &X, unsigned dx, unsigned dy, unsigned dz) {
    return autodiff3rd(
     [](autodiff::dual3rd x, autodiff::dual3rd y, autodiff::dual3rd z)->autodiff::dual3rd {
      return (3.*x-.5)*(5.*y+2.)*(z-0.7);
     },X,dx,dy,dz);
  }
};
struct P2scalar {
  static double f(const Eigen::Vector3d &X, unsigned dx, unsigned dy, unsigned dz) {
    return autodiff3rd(
     [](autodiff::dual3rd x, autodiff::dual3rd y, autodiff::dual3rd z)->autodiff::dual3rd {
      return (3.*x*x+x-.5)*(y*y-5.*y+2.)*(2.*z*z+z-0.7);
     },X,dx,dy,dz);
  }
};
struct P3scalar {
  static double f(const Eigen::Vector3d &X, unsigned dx, unsigned dy, unsigned dz) {
    return autodiff3rd(
     [](autodiff::dual3rd x, autodiff::dual3rd y, autodiff::dual3rd z)->autodiff::dual3rd {
      return (0.5*x*x*x+3.*x*x+x-.5)*(0.4*y*y*y-y*y-5.*y+2.)*(z*z*z+2.*z*z+z-0.7);
     },X,dx,dy,dz);
  }
};
#endif
