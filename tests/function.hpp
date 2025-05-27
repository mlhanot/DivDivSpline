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

/// Wrapper for third order derivative
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
struct P0form {
  static double f(unsigned i, const Eigen::Vector3d &X, unsigned dx, unsigned dy, unsigned dz) {
    if (i == 0) {
      return autodiff1st(
        [](autodiff::real x, autodiff::real y, autodiff::real z)->autodiff::real {
          return (autodiff::detail::pow(y, 2) + 3*y - 1)*(2*autodiff::detail::pow(z, 2) - z + 2)*(3*autodiff::detail::pow(x, 3) + 2*autodiff::detail::pow(x, 2) - x + 1);
        },X,dx,dy,dz);
    } else if (i == 1) {
      return autodiff1st(
        [](autodiff::real x, autodiff::real y, autodiff::real z)->autodiff::real {
          return (5*autodiff::detail::pow(x, 2) - x + 1)*(3*autodiff::detail::pow(z, 2) - z + 3)*(4*autodiff::detail::pow(y, 3) - autodiff::detail::pow(y, 2) + 2*y - 1);
        },X,dx,dy,dz);
    } else {
      return autodiff1st(
        [](autodiff::real x, autodiff::real y, autodiff::real z)->autodiff::real {
          return (autodiff::detail::pow(x, 2) - x + 4)*(autodiff::detail::pow(y, 2) - 3*y + 1)*(autodiff::detail::pow(z, 3) + 2*autodiff::detail::pow(z, 2) + z - 4);
        },X,dx,dy,dz);
    };
  }
  static double df(unsigned i, unsigned j, const Eigen::Vector3d &X, unsigned dx, unsigned dy, unsigned dz) {
    if (i == 0 && j == 0) {
      return autodiff1st(
        [](autodiff::real x, autodiff::real y, autodiff::real z)->autodiff::real {
          return -1.0/3.0*(autodiff::detail::pow(x, 2) - x + 4)*(autodiff::detail::pow(y, 2) - 3*y + 1)*(3*autodiff::detail::pow(z, 2) + 4*z + 1) - 2.0/3.0*(5*autodiff::detail::pow(x, 2) - x + 1)*(6*autodiff::detail::pow(y, 2) - y + 1)*(3*autodiff::detail::pow(z, 2) - z + 3) + (2.0/3.0)*(9*autodiff::detail::pow(x, 2) + 4*x - 1)*(autodiff::detail::pow(y, 2) + 3*y - 1)*(2*autodiff::detail::pow(z, 2) - z + 2);
        },X,dx,dy,dz);
    } else if (i == 0 && j == 1) {
      return autodiff1st(
        [](autodiff::real x, autodiff::real y, autodiff::real z)->autodiff::real {
          return (2*y + 3)*(2*autodiff::detail::pow(z, 2) - z + 2)*(3*autodiff::detail::pow(x, 3) + 2*autodiff::detail::pow(x, 2) - x + 1);
        },X,dx,dy,dz);
    } else if (i == 0 && j == 2) {
      return autodiff1st(
        [](autodiff::real x, autodiff::real y, autodiff::real z)->autodiff::real {
          return (4*z - 1)*(autodiff::detail::pow(y, 2) + 3*y - 1)*(3*autodiff::detail::pow(x, 3) + 2*autodiff::detail::pow(x, 2) - x + 1);
        },X,dx,dy,dz);
    } else if (i == 1 && j == 0) {
      return autodiff1st(
        [](autodiff::real x, autodiff::real y, autodiff::real z)->autodiff::real {
          return (10*x - 1)*(3*autodiff::detail::pow(z, 2) - z + 3)*(4*autodiff::detail::pow(y, 3) - autodiff::detail::pow(y, 2) + 2*y - 1);
        },X,dx,dy,dz);
    } else if (i == 1 && j == 1) {
      return autodiff1st(
        [](autodiff::real x, autodiff::real y, autodiff::real z)->autodiff::real {
          return -1.0/3.0*(autodiff::detail::pow(x, 2) - x + 4)*(autodiff::detail::pow(y, 2) - 3*y + 1)*(3*autodiff::detail::pow(z, 2) + 4*z + 1) + (4.0/3.0)*(5*autodiff::detail::pow(x, 2) - x + 1)*(6*autodiff::detail::pow(y, 2) - y + 1)*(3*autodiff::detail::pow(z, 2) - z + 3) - 1.0/3.0*(9*autodiff::detail::pow(x, 2) + 4*x - 1)*(autodiff::detail::pow(y, 2) + 3*y - 1)*(2*autodiff::detail::pow(z, 2) - z + 2);
        },X,dx,dy,dz);
    } else if (i == 1 && j == 2) {
      return autodiff1st(
        [](autodiff::real x, autodiff::real y, autodiff::real z)->autodiff::real {
          return (6*z - 1)*(5*autodiff::detail::pow(x, 2) - x + 1)*(4*autodiff::detail::pow(y, 3) - autodiff::detail::pow(y, 2) + 2*y - 1);
        },X,dx,dy,dz);
    } else if (i == 2 && j == 0) {
      return autodiff1st(
        [](autodiff::real x, autodiff::real y, autodiff::real z)->autodiff::real {
          return (2*x - 1)*(autodiff::detail::pow(y, 2) - 3*y + 1)*(autodiff::detail::pow(z, 3) + 2*autodiff::detail::pow(z, 2) + z - 4);
        },X,dx,dy,dz);
    } else if (i == 2 && j == 1) {
      return autodiff1st(
        [](autodiff::real x, autodiff::real y, autodiff::real z)->autodiff::real {
          return (2*y - 3)*(autodiff::detail::pow(x, 2) - x + 4)*(autodiff::detail::pow(z, 3) + 2*autodiff::detail::pow(z, 2) + z - 4);
        },X,dx,dy,dz);
    } else {
      return autodiff1st(
        [](autodiff::real x, autodiff::real y, autodiff::real z)->autodiff::real {
          return (2.0/3.0)*(autodiff::detail::pow(x, 2) - x + 4)*(autodiff::detail::pow(y, 2) - 3*y + 1)*(3*autodiff::detail::pow(z, 2) + 4*z + 1) - 2.0/3.0*(5*autodiff::detail::pow(x, 2) - x + 1)*(6*autodiff::detail::pow(y, 2) - y + 1)*(3*autodiff::detail::pow(z, 2) - z + 3) - 1.0/3.0*(9*autodiff::detail::pow(x, 2) + 4*x - 1)*(autodiff::detail::pow(y, 2) + 3*y - 1)*(2*autodiff::detail::pow(z, 2) - z + 2);
        },X,dx,dy,dz);
    };
  }
};
struct P1form {
  static double f(unsigned i, unsigned j, const Eigen::Vector3d &X, unsigned dx, unsigned dy, unsigned dz) {
    if (i == 0 && j == 0) {
      return autodiff1st(
        [](autodiff::real x, autodiff::real y, autodiff::real z)->autodiff::real {
          return (2*autodiff::detail::pow(x, 2) - x + 1)*(autodiff::detail::pow(y, 2) + 3*y - 1)*(2*autodiff::detail::pow(z, 2) - z + 2);
        },X,dx,dy,dz);
    } else if (i == 0 && j == 1) {
      return autodiff1st(
        [](autodiff::real x, autodiff::real y, autodiff::real z)->autodiff::real {
          return (3*y - 1)*(2*autodiff::detail::pow(z, 2) - z + 2)*(autodiff::detail::pow(x, 3) - 2*autodiff::detail::pow(x, 2) - x + 1);
        },X,dx,dy,dz);
    } else if (i == 0 && j == 2) {
      return autodiff1st(
        [](autodiff::real x, autodiff::real y, autodiff::real z)->autodiff::real {
          return (z + 2)*(autodiff::detail::pow(y, 2) + 3*y - 1)*(2*autodiff::detail::pow(x, 3) + autodiff::detail::pow(x, 2) - x + 1);
        },X,dx,dy,dz);
    } else if (i == 1 && j == 0) {
      return autodiff1st(
        [](autodiff::real x, autodiff::real y, autodiff::real z)->autodiff::real {
          return (x + 1)*(3*autodiff::detail::pow(z, 2) - z + 3)*(4*autodiff::detail::pow(y, 3) - autodiff::detail::pow(y, 2) + 2*y - 1);
        },X,dx,dy,dz);
    } else if (i == 1 && j == 1) {
      return autodiff1st(
        [](autodiff::real x, autodiff::real y, autodiff::real z)->autodiff::real {
          return -(3*autodiff::detail::pow(x, 2) - x + 1)*(autodiff::detail::pow(y, 2) - y + 1)*(3*autodiff::detail::pow(z, 2) - z + 3);
        },X,dx,dy,dz);
    } else if (i == 1 && j == 2) {
      return autodiff1st(
        [](autodiff::real x, autodiff::real y, autodiff::real z)->autodiff::real {
          return (z - 1)*(5*autodiff::detail::pow(x, 2) - x + 1)*(4*autodiff::detail::pow(y, 3) - autodiff::detail::pow(y, 2) + 2*y - 1);
        },X,dx,dy,dz);
    } else if (i == 2 && j == 0) {
      return autodiff1st(
        [](autodiff::real x, autodiff::real y, autodiff::real z)->autodiff::real {
          return -(x - 4)*(autodiff::detail::pow(y, 2) - 3*y + 1)*(autodiff::detail::pow(z, 3) + 2*autodiff::detail::pow(z, 2) + z - 3);
        },X,dx,dy,dz);
    } else if (i == 2 && j == 1) {
      return autodiff1st(
        [](autodiff::real x, autodiff::real y, autodiff::real z)->autodiff::real {
          return -(y - 1)*(2*autodiff::detail::pow(x, 2) - x + 4)*(autodiff::detail::pow(z, 3) + 2*autodiff::detail::pow(z, 2) + z - 4);
        },X,dx,dy,dz);
    } else {
      return autodiff1st(
        [](autodiff::real x, autodiff::real y, autodiff::real z)->autodiff::real {
          return -(2*autodiff::detail::pow(x, 2) - x + 1)*(autodiff::detail::pow(y, 2) + 3*y - 1)*(2*autodiff::detail::pow(z, 2) - z + 2) + (3*autodiff::detail::pow(x, 2) - x + 1)*(autodiff::detail::pow(y, 2) - y + 1)*(3*autodiff::detail::pow(z, 2) - z + 3);
        },X,dx,dy,dz);
    };
  }
  static double df(unsigned i, unsigned j, const Eigen::Vector3d &X, unsigned dx, unsigned dy, unsigned dz) {
    if (i == 0 && j == 0) {
      return autodiff1st(
        [](autodiff::real x, autodiff::real y, autodiff::real z)->autodiff::real {
          return (2*y + 3)*(z + 2)*(2*autodiff::detail::pow(x, 3) + autodiff::detail::pow(x, 2) - x + 1) - (3*y - 1)*(4*z - 1)*(autodiff::detail::pow(x, 3) - 2*autodiff::detail::pow(x, 2) - x + 1);
        },X,dx,dy,dz);
    } else if (i == 0 && j == 1) {
      return autodiff1st(
        [](autodiff::real x, autodiff::real y, autodiff::real z)->autodiff::real {
          return (z - 1)*(5*autodiff::detail::pow(x, 2) - x + 1)*(6*autodiff::detail::pow(y, 2) - y + 1) - 1.0/2.0*(z + 2)*(6*autodiff::detail::pow(x, 2) + 2*x - 1)*(autodiff::detail::pow(y, 2) + 3*y - 1) + (1.0/2.0)*(4*z - 1)*(2*autodiff::detail::pow(x, 2) - x + 1)*(autodiff::detail::pow(y, 2) + 3*y - 1) + (1.0/2.0)*(6*z - 1)*(3*autodiff::detail::pow(x, 2) - x + 1)*(autodiff::detail::pow(y, 2) - y + 1);
        },X,dx,dy,dz);
    } else if (i == 0 && j == 2) {
      return autodiff1st(
        [](autodiff::real x, autodiff::real y, autodiff::real z)->autodiff::real {
          return (1.0/2.0)*(y - 1)*(2*autodiff::detail::pow(x, 2) - x + 4)*(3*autodiff::detail::pow(z, 2) + 4*z + 1) + (1.0/2.0)*(2*y - 1)*(3*autodiff::detail::pow(x, 2) - x + 1)*(3*autodiff::detail::pow(z, 2) - z + 3) - (2*y + 3)*(2*autodiff::detail::pow(x, 2) - x + 1)*(2*autodiff::detail::pow(z, 2) - z + 2) - 1.0/2.0*(3*y - 1)*(-3*autodiff::detail::pow(x, 2) + 4*x + 1)*(2*autodiff::detail::pow(z, 2) - z + 2);
        },X,dx,dy,dz);
    } else if (i == 1 && j == 0) {
      return autodiff1st(
        [](autodiff::real x, autodiff::real y, autodiff::real z)->autodiff::real {
          return (z - 1)*(5*autodiff::detail::pow(x, 2) - x + 1)*(6*autodiff::detail::pow(y, 2) - y + 1) - 1.0/2.0*(z + 2)*(6*autodiff::detail::pow(x, 2) + 2*x - 1)*(autodiff::detail::pow(y, 2) + 3*y - 1) + (1.0/2.0)*(4*z - 1)*(2*autodiff::detail::pow(x, 2) - x + 1)*(autodiff::detail::pow(y, 2) + 3*y - 1) + (1.0/2.0)*(6*z - 1)*(3*autodiff::detail::pow(x, 2) - x + 1)*(autodiff::detail::pow(y, 2) - y + 1);
        },X,dx,dy,dz);
    } else if (i == 1 && j == 1) {
      return autodiff1st(
        [](autodiff::real x, autodiff::real y, autodiff::real z)->autodiff::real {
          return ((x + 1)*(6*z - 1) - (10*x - 1)*(z - 1))*(4*autodiff::detail::pow(y, 3) - autodiff::detail::pow(y, 2) + 2*y - 1);
        },X,dx,dy,dz);
    } else if (i == 1 && j == 2) {
      return autodiff1st(
        [](autodiff::real x, autodiff::real y, autodiff::real z)->autodiff::real {
          return -1.0/2.0*(x - 4)*(autodiff::detail::pow(y, 2) - 3*y + 1)*(3*autodiff::detail::pow(z, 2) + 4*z + 1) - (x + 1)*(6*autodiff::detail::pow(y, 2) - y + 1)*(3*autodiff::detail::pow(z, 2) - z + 3) + (1.0/2.0)*(4*x - 1)*(autodiff::detail::pow(y, 2) + 3*y - 1)*(2*autodiff::detail::pow(z, 2) - z + 2) - (6*x - 1)*(autodiff::detail::pow(y, 2) - y + 1)*(3*autodiff::detail::pow(z, 2) - z + 3);
        },X,dx,dy,dz);
    } else if (i == 2 && j == 0) {
      return autodiff1st(
        [](autodiff::real x, autodiff::real y, autodiff::real z)->autodiff::real {
          return (1.0/2.0)*(y - 1)*(2*autodiff::detail::pow(x, 2) - x + 4)*(3*autodiff::detail::pow(z, 2) + 4*z + 1) + (1.0/2.0)*(2*y - 1)*(3*autodiff::detail::pow(x, 2) - x + 1)*(3*autodiff::detail::pow(z, 2) - z + 3) - (2*y + 3)*(2*autodiff::detail::pow(x, 2) - x + 1)*(2*autodiff::detail::pow(z, 2) - z + 2) - 1.0/2.0*(3*y - 1)*(-3*autodiff::detail::pow(x, 2) + 4*x + 1)*(2*autodiff::detail::pow(z, 2) - z + 2);
        },X,dx,dy,dz);
    } else if (i == 2 && j == 1) {
      return autodiff1st(
        [](autodiff::real x, autodiff::real y, autodiff::real z)->autodiff::real {
          return -1.0/2.0*(x - 4)*(autodiff::detail::pow(y, 2) - 3*y + 1)*(3*autodiff::detail::pow(z, 2) + 4*z + 1) - (x + 1)*(6*autodiff::detail::pow(y, 2) - y + 1)*(3*autodiff::detail::pow(z, 2) - z + 3) + (1.0/2.0)*(4*x - 1)*(autodiff::detail::pow(y, 2) + 3*y - 1)*(2*autodiff::detail::pow(z, 2) - z + 2) - (6*x - 1)*(autodiff::detail::pow(y, 2) - y + 1)*(3*autodiff::detail::pow(z, 2) - z + 3);
        },X,dx,dy,dz);
    } else {
      return autodiff1st(
        [](autodiff::real x, autodiff::real y, autodiff::real z)->autodiff::real {
          return (x - 4)*(2*y - 3)*(autodiff::detail::pow(z, 3) + 2*autodiff::detail::pow(z, 2) + z - 3) - (4*x - 1)*(y - 1)*(autodiff::detail::pow(z, 3) + 2*autodiff::detail::pow(z, 2) + z - 4);
        },X,dx,dy,dz);
    };
  }
};
struct P2form {
  static double f(unsigned i, unsigned j, const Eigen::Vector3d &X, unsigned dx, unsigned dy, unsigned dz) {
    if (i == 0 && j == 0) {
      return autodiff1st(
        [](autodiff::real x, autodiff::real y, autodiff::real z)->autodiff::real {
          return (3*y - 1)*(z + 2)*(3*autodiff::detail::pow(x, 3) + 2*autodiff::detail::pow(x, 2) - x + 1);
        },X,dx,dy,dz);
    } else if (i == 0 && j == 1) {
      return autodiff1st(
        [](autodiff::real x, autodiff::real y, autodiff::real z)->autodiff::real {
          return -(z - 2)*(2*autodiff::detail::pow(x, 2) - x + 1)*(autodiff::detail::pow(y, 2) + 3*y - 1);
        },X,dx,dy,dz);
    } else if (i == 0 && j == 2) {
      return autodiff1st(
        [](autodiff::real x, autodiff::real y, autodiff::real z)->autodiff::real {
          return (y + 1)*(autodiff::detail::pow(x, 2) - x + 1)*(2*autodiff::detail::pow(z, 2) - z + 2);
        },X,dx,dy,dz);
    } else if (i == 1 && j == 0) {
      return autodiff1st(
        [](autodiff::real x, autodiff::real y, autodiff::real z)->autodiff::real {
          return -(z - 2)*(2*autodiff::detail::pow(x, 2) - x + 1)*(autodiff::detail::pow(y, 2) + 3*y - 1);
        },X,dx,dy,dz);
    } else if (i == 1 && j == 1) {
      return autodiff1st(
        [](autodiff::real x, autodiff::real y, autodiff::real z)->autodiff::real {
          return (x + 1)*(z + 1)*(4*autodiff::detail::pow(y, 3) - autodiff::detail::pow(y, 2) + 2*y - 1);
        },X,dx,dy,dz);
    } else if (i == 1 && j == 2) {
      return autodiff1st(
        [](autodiff::real x, autodiff::real y, autodiff::real z)->autodiff::real {
          return -(x - 4)*(autodiff::detail::pow(y, 2) - 3*y + 1)*(2*autodiff::detail::pow(z, 2) + z - 4);
        },X,dx,dy,dz);
    } else if (i == 2 && j == 0) {
      return autodiff1st(
        [](autodiff::real x, autodiff::real y, autodiff::real z)->autodiff::real {
          return (y + 1)*(autodiff::detail::pow(x, 2) - x + 1)*(2*autodiff::detail::pow(z, 2) - z + 2);
        },X,dx,dy,dz);
    } else if (i == 2 && j == 1) {
      return autodiff1st(
        [](autodiff::real x, autodiff::real y, autodiff::real z)->autodiff::real {
          return -(x - 4)*(autodiff::detail::pow(y, 2) - 3*y + 1)*(2*autodiff::detail::pow(z, 2) + z - 4);
        },X,dx,dy,dz);
    } else {
      return autodiff1st(
        [](autodiff::real x, autodiff::real y, autodiff::real z)->autodiff::real {
          return -(x + 4)*(y - 1)*(autodiff::detail::pow(z, 3) + 2*autodiff::detail::pow(z, 2) + z - 4);
        },X,dx,dy,dz);
    };
  }
  static double df(const Eigen::Vector3d &X, unsigned dx, unsigned dy, unsigned dz) {
      return autodiff1st(
        [](autodiff::real x, autodiff::real y, autodiff::real z)->autodiff::real {
          return 56*x*y*z + 152*x*y + 2*x*z + 16*x + 72*y*z + 42*y - 80*z - 28;
        },X,dx,dy,dz);
;
  }
};
struct P3form {
  static double f(const Eigen::Vector3d &X, unsigned dx, unsigned dy, unsigned dz) {
      return autodiff1st(
        [](autodiff::real x, autodiff::real y, autodiff::real z)->autodiff::real {
          return 57*x*y*z + 151*x*y + 2*x*z + 16*x + 72*y*z + 42*y - 80*z - 28;
        },X,dx,dy,dz);
;
  }
  static double df(const Eigen::Vector3d &X, unsigned dx, unsigned dy, unsigned dz) {
      return 0.;
;
  }
};
struct P0formAlt {
  static double f(unsigned i, const Eigen::Vector3d &X, unsigned dx, unsigned dy, unsigned dz) {
    if (i == 0) {
      return autodiff1st(
        [](autodiff::real x, autodiff::real y, autodiff::real z)->autodiff::real {
          return (x - 2)*(y - 3)*(z - 4);
        },X,dx,dy,dz);
    } else if (i == 1) {
      return autodiff1st(
        [](autodiff::real x, autodiff::real y, autodiff::real z)->autodiff::real {
          return (3*x + 1)*(2*y - 1)*(z + 1);
        },X,dx,dy,dz);
    } else {
      return autodiff1st(
        [](autodiff::real x, autodiff::real y, autodiff::real z)->autodiff::real {
          return (x + 1)*(y - 1)*(z + 3);
        },X,dx,dy,dz);
    };
  }
  static double df(unsigned i, unsigned j, const Eigen::Vector3d &X, unsigned dx, unsigned dy, unsigned dz) {
    if (i == 0 && j == 0) {
      return autodiff1st(
        [](autodiff::real x, autodiff::real y, autodiff::real z)->autodiff::real {
          return -1.0/3.0*(x + 1)*(y - 1) - 2.0/3.0*(3*x + 1)*(z + 1) + (2.0/3.0)*(y - 3)*(z - 4);
        },X,dx,dy,dz);
    } else if (i == 0 && j == 1) {
      return autodiff1st(
        [](autodiff::real x, autodiff::real y, autodiff::real z)->autodiff::real {
          return (x - 2)*(z - 4);
        },X,dx,dy,dz);
    } else if (i == 0 && j == 2) {
      return autodiff1st(
        [](autodiff::real x, autodiff::real y, autodiff::real z)->autodiff::real {
          return (x - 2)*(y - 3);
        },X,dx,dy,dz);
    } else if (i == 1 && j == 0) {
      return autodiff1st(
        [](autodiff::real x, autodiff::real y, autodiff::real z)->autodiff::real {
          return 3*(2*y - 1)*(z + 1);
        },X,dx,dy,dz);
    } else if (i == 1 && j == 1) {
      return autodiff1st(
        [](autodiff::real x, autodiff::real y, autodiff::real z)->autodiff::real {
          return -1.0/3.0*(x + 1)*(y - 1) + (4.0/3.0)*(3*x + 1)*(z + 1) - 1.0/3.0*(y - 3)*(z - 4);
        },X,dx,dy,dz);
    } else if (i == 1 && j == 2) {
      return autodiff1st(
        [](autodiff::real x, autodiff::real y, autodiff::real z)->autodiff::real {
          return (3*x + 1)*(2*y - 1);
        },X,dx,dy,dz);
    } else if (i == 2 && j == 0) {
      return autodiff1st(
        [](autodiff::real x, autodiff::real y, autodiff::real z)->autodiff::real {
          return (y - 1)*(z + 3);
        },X,dx,dy,dz);
    } else if (i == 2 && j == 1) {
      return autodiff1st(
        [](autodiff::real x, autodiff::real y, autodiff::real z)->autodiff::real {
          return (x + 1)*(z + 3);
        },X,dx,dy,dz);
    } else {
      return autodiff1st(
        [](autodiff::real x, autodiff::real y, autodiff::real z)->autodiff::real {
          return (2.0/3.0)*x*y - 2*x*z - 8.0/3.0*x - 1.0/3.0*y*z + 2*y + (1.0/3.0)*z - 16.0/3.0;
        },X,dx,dy,dz);
    };
  }
};
struct P1formAlt {
  static double f(unsigned i, unsigned j, const Eigen::Vector3d &X, unsigned dx, unsigned dy, unsigned dz) {
    if (i == 0 && j == 0) {
      return autodiff1st(
        [](autodiff::real x, autodiff::real y, autodiff::real z)->autodiff::real {
          return (x - 2)*(y - 3)*(z - 4);
        },X,dx,dy,dz);
    } else if (i == 0 && j == 1) {
      return autodiff1st(
        [](autodiff::real x, autodiff::real y, autodiff::real z)->autodiff::real {
          return (x - 2)*(y - 3)*(z - 4);
        },X,dx,dy,dz);
    } else if (i == 0 && j == 2) {
      return autodiff1st(
        [](autodiff::real x, autodiff::real y, autodiff::real z)->autodiff::real {
          return (x - 2)*(y - 3)*(z - 4);
        },X,dx,dy,dz);
    } else if (i == 1 && j == 0) {
      return autodiff1st(
        [](autodiff::real x, autodiff::real y, autodiff::real z)->autodiff::real {
          return (3*x + 1)*(2*y - 1)*(z + 1);
        },X,dx,dy,dz);
    } else if (i == 1 && j == 1) {
      return autodiff1st(
        [](autodiff::real x, autodiff::real y, autodiff::real z)->autodiff::real {
          return (x + 1)*(y - 1)*(z + 3);
        },X,dx,dy,dz);
    } else if (i == 1 && j == 2) {
      return autodiff1st(
        [](autodiff::real x, autodiff::real y, autodiff::real z)->autodiff::real {
          return (x + 1)*(y - 1)*(z + 3);
        },X,dx,dy,dz);
    } else if (i == 2 && j == 0) {
      return autodiff1st(
        [](autodiff::real x, autodiff::real y, autodiff::real z)->autodiff::real {
          return (x + 1)*(y - 1)*(z + 3);
        },X,dx,dy,dz);
    } else if (i == 2 && j == 1) {
      return autodiff1st(
        [](autodiff::real x, autodiff::real y, autodiff::real z)->autodiff::real {
          return (3*x + 1)*(2*y - 1)*(z + 1);
        },X,dx,dy,dz);
    } else {
      return autodiff1st(
        [](autodiff::real x, autodiff::real y, autodiff::real z)->autodiff::real {
          return -(x - 2)*(y - 3)*(z - 4) - (x + 1)*(y - 1)*(z + 3);
        },X,dx,dy,dz);
    };
  }
  static double df(unsigned i, unsigned j, const Eigen::Vector3d &X, unsigned dx, unsigned dy, unsigned dz) {
    if (i == 0 && j == 0) {
      return autodiff1st(
        [](autodiff::real x, autodiff::real y, autodiff::real z)->autodiff::real {
          return (x - 2)*(-y + z - 1);
        },X,dx,dy,dz);
    } else if (i == 0 && j == 1) {
      return autodiff1st(
        [](autodiff::real x, autodiff::real y, autodiff::real z)->autodiff::real {
          return (1.0/2.0)*x*z + (1.0/2.0)*x - 1.0/2.0*y*z + (1.0/2.0)*y + 2*z - 1;
        },X,dx,dy,dz);
    } else if (i == 0 && j == 2) {
      return autodiff1st(
        [](autodiff::real x, autodiff::real y, autodiff::real z)->autodiff::real {
          return -3*x*y - 3.0/2.0*x*z + 4*x + (1.0/2.0)*y*z - 3*y - 3;
        },X,dx,dy,dz);
    } else if (i == 1 && j == 0) {
      return autodiff1st(
        [](autodiff::real x, autodiff::real y, autodiff::real z)->autodiff::real {
          return (1.0/2.0)*x*z + (1.0/2.0)*x - 1.0/2.0*y*z + (1.0/2.0)*y + 2*z - 1;
        },X,dx,dy,dz);
    } else if (i == 1 && j == 1) {
      return autodiff1st(
        [](autodiff::real x, autodiff::real y, autodiff::real z)->autodiff::real {
          return (3*x + 1)*(2*y - 1) - (y - 1)*(z + 3);
        },X,dx,dy,dz);
    } else if (i == 1 && j == 2) {
      return autodiff1st(
        [](autodiff::real x, autodiff::real y, autodiff::real z)->autodiff::real {
          return (1.0/2.0)*x*y - 3*x*z - 7.0/2.0*x + (3.0/2.0)*y*z + (3.0/2.0)*y - 7.0/2.0*z + 3.0/2.0;
        },X,dx,dy,dz);
    } else if (i == 2 && j == 0) {
      return autodiff1st(
        [](autodiff::real x, autodiff::real y, autodiff::real z)->autodiff::real {
          return -3*x*y - 3.0/2.0*x*z + 4*x + (1.0/2.0)*y*z - 3*y - 3;
        },X,dx,dy,dz);
    } else if (i == 2 && j == 1) {
      return autodiff1st(
        [](autodiff::real x, autodiff::real y, autodiff::real z)->autodiff::real {
          return (1.0/2.0)*x*y - 3*x*z - 7.0/2.0*x + (3.0/2.0)*y*z + (3.0/2.0)*y - 7.0/2.0*z + 3.0/2.0;
        },X,dx,dy,dz);
    } else {
      return autodiff1st(
        [](autodiff::real x, autodiff::real y, autodiff::real z)->autodiff::real {
          return -(x + 1)*(z + 3) + 3*(2*y - 1)*(z + 1);
        },X,dx,dy,dz);
    };
  }
};
struct P2formAlt {
  static double f(unsigned i, unsigned j, const Eigen::Vector3d &X, unsigned dx, unsigned dy, unsigned dz) {
    if (i == 0 && j == 0) {
      return autodiff1st(
        [](autodiff::real x, autodiff::real y, autodiff::real z)->autodiff::real {
          return (x - 2)*(y - 3)*(z - 4);
        },X,dx,dy,dz);
    } else if (i == 0 && j == 1) {
      return autodiff1st(
        [](autodiff::real x, autodiff::real y, autodiff::real z)->autodiff::real {
          return (3*x + 1)*(2*y - 1)*(z + 1);
        },X,dx,dy,dz);
    } else if (i == 0 && j == 2) {
      return autodiff1st(
        [](autodiff::real x, autodiff::real y, autodiff::real z)->autodiff::real {
          return (x - 2)*(y - 3)*(z - 4);
        },X,dx,dy,dz);
    } else if (i == 1 && j == 0) {
      return autodiff1st(
        [](autodiff::real x, autodiff::real y, autodiff::real z)->autodiff::real {
          return (3*x + 1)*(2*y - 1)*(z + 1);
        },X,dx,dy,dz);
    } else if (i == 1 && j == 1) {
      return autodiff1st(
        [](autodiff::real x, autodiff::real y, autodiff::real z)->autodiff::real {
          return (x + 1)*(y - 1)*(z + 3);
        },X,dx,dy,dz);
    } else if (i == 1 && j == 2) {
      return autodiff1st(
        [](autodiff::real x, autodiff::real y, autodiff::real z)->autodiff::real {
          return (3*x + 1)*(2*y - 1)*(z + 1);
        },X,dx,dy,dz);
    } else if (i == 2 && j == 0) {
      return autodiff1st(
        [](autodiff::real x, autodiff::real y, autodiff::real z)->autodiff::real {
          return (x - 2)*(y - 3)*(z - 4);
        },X,dx,dy,dz);
    } else if (i == 2 && j == 1) {
      return autodiff1st(
        [](autodiff::real x, autodiff::real y, autodiff::real z)->autodiff::real {
          return (3*x + 1)*(2*y - 1)*(z + 1);
        },X,dx,dy,dz);
    } else {
      return autodiff1st(
        [](autodiff::real x, autodiff::real y, autodiff::real z)->autodiff::real {
          return (x + 4)*(y - 1)*(z + 1);
        },X,dx,dy,dz);
    };
  }
  static double df(const Eigen::Vector3d &X, unsigned dx, unsigned dy, unsigned dz) {
      return autodiff1st(
        [](autodiff::real x, autodiff::real y, autodiff::real z)->autodiff::real {
          return 12*x + 2*y + 12*z + 10;
        },X,dx,dy,dz);
;
  }
};
struct P3formAlt {
  static double f(const Eigen::Vector3d &X, unsigned dx, unsigned dy, unsigned dz) {
      return autodiff1st(
        [](autodiff::real x, autodiff::real y, autodiff::real z)->autodiff::real {
          return 5*x*y*z + 146*x*y + 3*x*z + 6*x + 7*y*z + 2*y - 3*z - 1;
        },X,dx,dy,dz);
;
  }
  static double df(const Eigen::Vector3d &X, unsigned dx, unsigned dy, unsigned dz) {
      return 0.;
;
  }
};

#endif
