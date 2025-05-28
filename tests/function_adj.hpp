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

struct P1C0 {
  static double f(unsigned i, unsigned j, const Eigen::Vector3d &X, unsigned dx, unsigned dy, unsigned dz) {
    if (i == 0 && j == 0) {
      return autodiff1st(
        [](autodiff::real x, autodiff::real y, autodiff::real z)->autodiff::real {
          return 0;
        },X,dx,dy,dz);
    } else if (i == 0 && j == 1) {
      return autodiff1st(
        [](autodiff::real x, autodiff::real y, autodiff::real z)->autodiff::real {
          return 1;
        },X,dx,dy,dz);
    } else if (i == 0 && j == 2) {
      return autodiff1st(
        [](autodiff::real x, autodiff::real y, autodiff::real z)->autodiff::real {
          return 2;
        },X,dx,dy,dz);
    } else if (i == 1 && j == 0) {
      return autodiff1st(
        [](autodiff::real x, autodiff::real y, autodiff::real z)->autodiff::real {
          return 0;
        },X,dx,dy,dz);
    } else if (i == 1 && j == 1) {
      return autodiff1st(
        [](autodiff::real x, autodiff::real y, autodiff::real z)->autodiff::real {
          return 0;
        },X,dx,dy,dz);
    } else if (i == 1 && j == 2) {
      return autodiff1st(
        [](autodiff::real x, autodiff::real y, autodiff::real z)->autodiff::real {
          return 0;
        },X,dx,dy,dz);
    } else if (i == 2 && j == 0) {
      return autodiff1st(
        [](autodiff::real x, autodiff::real y, autodiff::real z)->autodiff::real {
          return 0;
        },X,dx,dy,dz);
    } else if (i == 2 && j == 1) {
      return autodiff1st(
        [](autodiff::real x, autodiff::real y, autodiff::real z)->autodiff::real {
          return 0;
        },X,dx,dy,dz);
    } else {
      return autodiff1st(
        [](autodiff::real x, autodiff::real y, autodiff::real z)->autodiff::real {
          return 0;
        },X,dx,dy,dz);
    };
  }
  static double deltaf(unsigned i, const Eigen::Vector3d &X, unsigned dx, unsigned dy, unsigned dz) {
    if (i == 0) {
      return autodiff1st(
        [](autodiff::real x, autodiff::real y, autodiff::real z)->autodiff::real {
          return 0;
        },X,dx,dy,dz);
    } else if (i == 1) {
      return autodiff1st(
        [](autodiff::real x, autodiff::real y, autodiff::real z)->autodiff::real {
          return 0;
        },X,dx,dy,dz);
    } else {
      return autodiff1st(
        [](autodiff::real x, autodiff::real y, autodiff::real z)->autodiff::real {
          return 0;
        },X,dx,dy,dz);
    };
  }
};
struct P1C1 {
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
  static double deltaf(unsigned i, const Eigen::Vector3d &X, unsigned dx, unsigned dy, unsigned dz) {
    if (i == 0) {
      return autodiff1st(
        [](autodiff::real x, autodiff::real y, autodiff::real z)->autodiff::real {
          return (4*x - 1)*(autodiff::detail::pow(y, 2) + 3*y - 1)*(2*autodiff::detail::pow(z, 2) - z + 2) + (autodiff::detail::pow(y, 2) + 3*y - 1)*(2*autodiff::detail::pow(x, 3) + autodiff::detail::pow(x, 2) - x + 1) + 3*(2*autodiff::detail::pow(z, 2) - z + 2)*(autodiff::detail::pow(x, 3) - 2*autodiff::detail::pow(x, 2) - x + 1);
        },X,dx,dy,dz);
    } else if (i == 1) {
      return autodiff1st(
        [](autodiff::real x, autodiff::real y, autodiff::real z)->autodiff::real {
          return -(2*y - 1)*(3*autodiff::detail::pow(x, 2) - x + 1)*(3*autodiff::detail::pow(z, 2) - z + 3) + (5*autodiff::detail::pow(x, 2) - x + 1)*(4*autodiff::detail::pow(y, 3) - autodiff::detail::pow(y, 2) + 2*y - 1) + (3*autodiff::detail::pow(z, 2) - z + 3)*(4*autodiff::detail::pow(y, 3) - autodiff::detail::pow(y, 2) + 2*y - 1);
        },X,dx,dy,dz);
    } else {
      return autodiff1st(
        [](autodiff::real x, autodiff::real y, autodiff::real z)->autodiff::real {
          return -(4*z - 1)*(2*autodiff::detail::pow(x, 2) - x + 1)*(autodiff::detail::pow(y, 2) + 3*y - 1) + (6*z - 1)*(3*autodiff::detail::pow(x, 2) - x + 1)*(autodiff::detail::pow(y, 2) - y + 1) - (2*autodiff::detail::pow(x, 2) - x + 4)*(autodiff::detail::pow(z, 3) + 2*autodiff::detail::pow(z, 2) + z - 4) - (autodiff::detail::pow(y, 2) - 3*y + 1)*(autodiff::detail::pow(z, 3) + 2*autodiff::detail::pow(z, 2) + z - 3);
        },X,dx,dy,dz);
    };
  }
};
struct P1C2 {
  static double f(unsigned i, unsigned j, const Eigen::Vector3d &X, unsigned dx, unsigned dy, unsigned dz) {
    if (i == 0 && j == 0) {
      return autodiff1st(
        [](autodiff::real x, autodiff::real y, autodiff::real z)->autodiff::real {
          return autodiff::detail::sin(y) + autodiff::detail::cos(x);
        },X,dx,dy,dz);
    } else if (i == 0 && j == 1) {
      return autodiff1st(
        [](autodiff::real x, autodiff::real y, autodiff::real z)->autodiff::real {
          return autodiff::detail::exp(z);
        },X,dx,dy,dz);
    } else if (i == 0 && j == 2) {
      return autodiff1st(
        [](autodiff::real x, autodiff::real y, autodiff::real z)->autodiff::real {
          return 2*x + 1;
        },X,dx,dy,dz);
    } else if (i == 1 && j == 0) {
      return autodiff1st(
        [](autodiff::real x, autodiff::real y, autodiff::real z)->autodiff::real {
          return autodiff::detail::sin(y);
        },X,dx,dy,dz);
    } else if (i == 1 && j == 1) {
      return autodiff1st(
        [](autodiff::real x, autodiff::real y, autodiff::real z)->autodiff::real {
          return -autodiff::detail::cos(x);
        },X,dx,dy,dz);
    } else if (i == 1 && j == 2) {
      return autodiff1st(
        [](autodiff::real x, autodiff::real y, autodiff::real z)->autodiff::real {
          return 0;
        },X,dx,dy,dz);
    } else if (i == 2 && j == 0) {
      return autodiff1st(
        [](autodiff::real x, autodiff::real y, autodiff::real z)->autodiff::real {
          return 0;
        },X,dx,dy,dz);
    } else if (i == 2 && j == 1) {
      return autodiff1st(
        [](autodiff::real x, autodiff::real y, autodiff::real z)->autodiff::real {
          return 0;
        },X,dx,dy,dz);
    } else {
      return autodiff1st(
        [](autodiff::real x, autodiff::real y, autodiff::real z)->autodiff::real {
          return -autodiff::detail::sin(y);
        },X,dx,dy,dz);
    };
  }
  static double deltaf(unsigned i, const Eigen::Vector3d &X, unsigned dx, unsigned dy, unsigned dz) {
    if (i == 0) {
      return autodiff1st(
        [](autodiff::real x, autodiff::real y, autodiff::real z)->autodiff::real {
          return -autodiff::detail::sin(x);
        },X,dx,dy,dz);
    } else if (i == 1) {
      return autodiff1st(
        [](autodiff::real x, autodiff::real y, autodiff::real z)->autodiff::real {
          return 0;
        },X,dx,dy,dz);
    } else {
      return autodiff1st(
        [](autodiff::real x, autodiff::real y, autodiff::real z)->autodiff::real {
          return 0;
        },X,dx,dy,dz);
    };
  }
};
struct P2C0 {
  static double f(unsigned i, unsigned j, const Eigen::Vector3d &X, unsigned dx, unsigned dy, unsigned dz) {
    if (i == 0 && j == 0) {
      return autodiff1st(
        [](autodiff::real x, autodiff::real y, autodiff::real z)->autodiff::real {
          return 0;
        },X,dx,dy,dz);
    } else if (i == 0 && j == 1) {
      return autodiff1st(
        [](autodiff::real x, autodiff::real y, autodiff::real z)->autodiff::real {
          return 1;
        },X,dx,dy,dz);
    } else if (i == 0 && j == 2) {
      return autodiff1st(
        [](autodiff::real x, autodiff::real y, autodiff::real z)->autodiff::real {
          return 2;
        },X,dx,dy,dz);
    } else if (i == 1 && j == 0) {
      return autodiff1st(
        [](autodiff::real x, autodiff::real y, autodiff::real z)->autodiff::real {
          return 1;
        },X,dx,dy,dz);
    } else if (i == 1 && j == 1) {
      return autodiff1st(
        [](autodiff::real x, autodiff::real y, autodiff::real z)->autodiff::real {
          return 0;
        },X,dx,dy,dz);
    } else if (i == 1 && j == 2) {
      return autodiff1st(
        [](autodiff::real x, autodiff::real y, autodiff::real z)->autodiff::real {
          return 0;
        },X,dx,dy,dz);
    } else if (i == 2 && j == 0) {
      return autodiff1st(
        [](autodiff::real x, autodiff::real y, autodiff::real z)->autodiff::real {
          return 2;
        },X,dx,dy,dz);
    } else if (i == 2 && j == 1) {
      return autodiff1st(
        [](autodiff::real x, autodiff::real y, autodiff::real z)->autodiff::real {
          return 0;
        },X,dx,dy,dz);
    } else {
      return autodiff1st(
        [](autodiff::real x, autodiff::real y, autodiff::real z)->autodiff::real {
          return 3;
        },X,dx,dy,dz);
    };
  }
  static double deltaf(unsigned i, unsigned j, const Eigen::Vector3d &X, unsigned dx, unsigned dy, unsigned dz) {
    if (i == 0 && j == 0) {
      return autodiff1st(
        [](autodiff::real x, autodiff::real y, autodiff::real z)->autodiff::real {
          return 0;
        },X,dx,dy,dz);
    } else if (i == 0 && j == 1) {
      return autodiff1st(
        [](autodiff::real x, autodiff::real y, autodiff::real z)->autodiff::real {
          return 0;
        },X,dx,dy,dz);
    } else if (i == 0 && j == 2) {
      return autodiff1st(
        [](autodiff::real x, autodiff::real y, autodiff::real z)->autodiff::real {
          return 0;
        },X,dx,dy,dz);
    } else if (i == 1 && j == 0) {
      return autodiff1st(
        [](autodiff::real x, autodiff::real y, autodiff::real z)->autodiff::real {
          return 0;
        },X,dx,dy,dz);
    } else if (i == 1 && j == 1) {
      return autodiff1st(
        [](autodiff::real x, autodiff::real y, autodiff::real z)->autodiff::real {
          return 0;
        },X,dx,dy,dz);
    } else if (i == 1 && j == 2) {
      return autodiff1st(
        [](autodiff::real x, autodiff::real y, autodiff::real z)->autodiff::real {
          return 0;
        },X,dx,dy,dz);
    } else if (i == 2 && j == 0) {
      return autodiff1st(
        [](autodiff::real x, autodiff::real y, autodiff::real z)->autodiff::real {
          return 0;
        },X,dx,dy,dz);
    } else if (i == 2 && j == 1) {
      return autodiff1st(
        [](autodiff::real x, autodiff::real y, autodiff::real z)->autodiff::real {
          return 0;
        },X,dx,dy,dz);
    } else {
      return autodiff1st(
        [](autodiff::real x, autodiff::real y, autodiff::real z)->autodiff::real {
          return 0;
        },X,dx,dy,dz);
    };
  }
};
struct P2C1 {
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
  static double deltaf(unsigned i, unsigned j, const Eigen::Vector3d &X, unsigned dx, unsigned dy, unsigned dz) {
    if (i == 0 && j == 0) {
      return autodiff1st(
        [](autodiff::real x, autodiff::real y, autodiff::real z)->autodiff::real {
          return (autodiff::detail::pow(x, 2) - x + 1)*(2*autodiff::detail::pow(z, 2) - z + 2) + (2*autodiff::detail::pow(x, 2) - x + 1)*(autodiff::detail::pow(y, 2) + 3*y - 1);
        },X,dx,dy,dz);
    } else if (i == 0 && j == 1) {
      return autodiff1st(
        [](autodiff::real x, autodiff::real y, autodiff::real z)->autodiff::real {
          return -(2*x - 1)*(y + 1)*(2*autodiff::detail::pow(z, 2) - z + 2) + (3*y - 1)*(3*autodiff::detail::pow(x, 3) + 2*autodiff::detail::pow(x, 2) - x + 1);
        },X,dx,dy,dz);
    } else if (i == 0 && j == 2) {
      return autodiff1st(
        [](autodiff::real x, autodiff::real y, autodiff::real z)->autodiff::real {
          return -(4*x - 1)*(z - 2)*(autodiff::detail::pow(y, 2) + 3*y - 1) - 3*(z + 2)*(3*autodiff::detail::pow(x, 3) + 2*autodiff::detail::pow(x, 2) - x + 1);
        },X,dx,dy,dz);
    } else if (i == 1 && j == 0) {
      return autodiff1st(
        [](autodiff::real x, autodiff::real y, autodiff::real z)->autodiff::real {
          return -(x - 4)*(2*y - 3)*(2*autodiff::detail::pow(z, 2) + z - 4) - (x + 1)*(4*autodiff::detail::pow(y, 3) - autodiff::detail::pow(y, 2) + 2*y - 1);
        },X,dx,dy,dz);
    } else if (i == 1 && j == 1) {
      return autodiff1st(
        [](autodiff::real x, autodiff::real y, autodiff::real z)->autodiff::real {
          return -(2*autodiff::detail::pow(x, 2) - x + 1)*(autodiff::detail::pow(y, 2) + 3*y - 1) + (autodiff::detail::pow(y, 2) - 3*y + 1)*(2*autodiff::detail::pow(z, 2) + z - 4);
        },X,dx,dy,dz);
    } else if (i == 1 && j == 2) {
      return autodiff1st(
        [](autodiff::real x, autodiff::real y, autodiff::real z)->autodiff::real {
          return (2*y + 3)*(z - 2)*(2*autodiff::detail::pow(x, 2) - x + 1) + (z + 1)*(4*autodiff::detail::pow(y, 3) - autodiff::detail::pow(y, 2) + 2*y - 1);
        },X,dx,dy,dz);
    } else if (i == 2 && j == 0) {
      return autodiff1st(
        [](autodiff::real x, autodiff::real y, autodiff::real z)->autodiff::real {
          return (x - 4)*(4*z + 1)*(autodiff::detail::pow(y, 2) - 3*y + 1) - (x + 4)*(autodiff::detail::pow(z, 3) + 2*autodiff::detail::pow(z, 2) + z - 4);
        },X,dx,dy,dz);
    } else if (i == 2 && j == 1) {
      return autodiff1st(
        [](autodiff::real x, autodiff::real y, autodiff::real z)->autodiff::real {
          return (y - 1)*(autodiff::detail::pow(z, 3) + 2*autodiff::detail::pow(z, 2) + z - 4) + (y + 1)*(4*z - 1)*(autodiff::detail::pow(x, 2) - x + 1);
        },X,dx,dy,dz);
    } else {
      return autodiff1st(
        [](autodiff::real x, autodiff::real y, autodiff::real z)->autodiff::real {
          return -(autodiff::detail::pow(x, 2) - x + 1)*(2*autodiff::detail::pow(z, 2) - z + 2) - (autodiff::detail::pow(y, 2) - 3*y + 1)*(2*autodiff::detail::pow(z, 2) + z - 4);
        },X,dx,dy,dz);
    };
  }
};
struct P2C2 {
  static double f(unsigned i, unsigned j, const Eigen::Vector3d &X, unsigned dx, unsigned dy, unsigned dz) {
    if (i == 0 && j == 0) {
      return autodiff1st(
        [](autodiff::real x, autodiff::real y, autodiff::real z)->autodiff::real {
          return autodiff::detail::sin(y) + autodiff::detail::cos(x);
        },X,dx,dy,dz);
    } else if (i == 0 && j == 1) {
      return autodiff1st(
        [](autodiff::real x, autodiff::real y, autodiff::real z)->autodiff::real {
          return autodiff::detail::exp(z);
        },X,dx,dy,dz);
    } else if (i == 0 && j == 2) {
      return autodiff1st(
        [](autodiff::real x, autodiff::real y, autodiff::real z)->autodiff::real {
          return 2*x + 1;
        },X,dx,dy,dz);
    } else if (i == 1 && j == 0) {
      return autodiff1st(
        [](autodiff::real x, autodiff::real y, autodiff::real z)->autodiff::real {
          return autodiff::detail::exp(z);
        },X,dx,dy,dz);
    } else if (i == 1 && j == 1) {
      return autodiff1st(
        [](autodiff::real x, autodiff::real y, autodiff::real z)->autodiff::real {
          return 0;
        },X,dx,dy,dz);
    } else if (i == 1 && j == 2) {
      return autodiff1st(
        [](autodiff::real x, autodiff::real y, autodiff::real z)->autodiff::real {
          return 0;
        },X,dx,dy,dz);
    } else if (i == 2 && j == 0) {
      return autodiff1st(
        [](autodiff::real x, autodiff::real y, autodiff::real z)->autodiff::real {
          return 2*x + 1;
        },X,dx,dy,dz);
    } else if (i == 2 && j == 1) {
      return autodiff1st(
        [](autodiff::real x, autodiff::real y, autodiff::real z)->autodiff::real {
          return 0;
        },X,dx,dy,dz);
    } else {
      return autodiff1st(
        [](autodiff::real x, autodiff::real y, autodiff::real z)->autodiff::real {
          return autodiff::detail::sin(y);
        },X,dx,dy,dz);
    };
  }
  static double deltaf(unsigned i, unsigned j, const Eigen::Vector3d &X, unsigned dx, unsigned dy, unsigned dz) {
    if (i == 0 && j == 0) {
      return autodiff1st(
        [](autodiff::real x, autodiff::real y, autodiff::real z)->autodiff::real {
          return -autodiff::detail::exp(z);
        },X,dx,dy,dz);
    } else if (i == 0 && j == 1) {
      return autodiff1st(
        [](autodiff::real x, autodiff::real y, autodiff::real z)->autodiff::real {
          return -2;
        },X,dx,dy,dz);
    } else if (i == 0 && j == 2) {
      return autodiff1st(
        [](autodiff::real x, autodiff::real y, autodiff::real z)->autodiff::real {
          return -autodiff::detail::cos(y);
        },X,dx,dy,dz);
    } else if (i == 1 && j == 0) {
      return autodiff1st(
        [](autodiff::real x, autodiff::real y, autodiff::real z)->autodiff::real {
          return 0;
        },X,dx,dy,dz);
    } else if (i == 1 && j == 1) {
      return autodiff1st(
        [](autodiff::real x, autodiff::real y, autodiff::real z)->autodiff::real {
          return autodiff::detail::exp(z);
        },X,dx,dy,dz);
    } else if (i == 1 && j == 2) {
      return autodiff1st(
        [](autodiff::real x, autodiff::real y, autodiff::real z)->autodiff::real {
          return 0;
        },X,dx,dy,dz);
    } else if (i == 2 && j == 0) {
      return autodiff1st(
        [](autodiff::real x, autodiff::real y, autodiff::real z)->autodiff::real {
          return autodiff::detail::cos(y);
        },X,dx,dy,dz);
    } else if (i == 2 && j == 1) {
      return autodiff1st(
        [](autodiff::real x, autodiff::real y, autodiff::real z)->autodiff::real {
          return 0;
        },X,dx,dy,dz);
    } else {
      return autodiff1st(
        [](autodiff::real x, autodiff::real y, autodiff::real z)->autodiff::real {
          return 0;
        },X,dx,dy,dz);
    };
  }
};
struct P3C0 {
  static double f(const Eigen::Vector3d &X, unsigned dx, unsigned dy, unsigned dz) {
      return autodiff1st(
        [](autodiff::real x, autodiff::real y, autodiff::real z)->autodiff::real {
          return 3;
        },X,dx,dy,dz);
;
  }
  static double deltaf(unsigned i, unsigned j, const Eigen::Vector3d &X, unsigned dx, unsigned dy, unsigned dz) {
    if (i == 0 && j == 0) {
      return autodiff1st(
        [](autodiff::real x, autodiff::real y, autodiff::real z)->autodiff::real {
          return 0;
        },X,dx,dy,dz);
    } else if (i == 0 && j == 1) {
      return autodiff1st(
        [](autodiff::real x, autodiff::real y, autodiff::real z)->autodiff::real {
          return 0;
        },X,dx,dy,dz);
    } else if (i == 0 && j == 2) {
      return autodiff1st(
        [](autodiff::real x, autodiff::real y, autodiff::real z)->autodiff::real {
          return 0;
        },X,dx,dy,dz);
    } else if (i == 1 && j == 0) {
      return autodiff1st(
        [](autodiff::real x, autodiff::real y, autodiff::real z)->autodiff::real {
          return 0;
        },X,dx,dy,dz);
    } else if (i == 1 && j == 1) {
      return autodiff1st(
        [](autodiff::real x, autodiff::real y, autodiff::real z)->autodiff::real {
          return 0;
        },X,dx,dy,dz);
    } else if (i == 1 && j == 2) {
      return autodiff1st(
        [](autodiff::real x, autodiff::real y, autodiff::real z)->autodiff::real {
          return 0;
        },X,dx,dy,dz);
    } else if (i == 2 && j == 0) {
      return autodiff1st(
        [](autodiff::real x, autodiff::real y, autodiff::real z)->autodiff::real {
          return 0;
        },X,dx,dy,dz);
    } else if (i == 2 && j == 1) {
      return autodiff1st(
        [](autodiff::real x, autodiff::real y, autodiff::real z)->autodiff::real {
          return 0;
        },X,dx,dy,dz);
    } else {
      return autodiff1st(
        [](autodiff::real x, autodiff::real y, autodiff::real z)->autodiff::real {
          return 0;
        },X,dx,dy,dz);
    };
  }
};
struct P3C1 {
  static double f(const Eigen::Vector3d &X, unsigned dx, unsigned dy, unsigned dz) {
      return autodiff1st(
        [](autodiff::real x, autodiff::real y, autodiff::real z)->autodiff::real {
          return 57*x*y*z + 151*x*y + 2*x*z + 16*x + 72*y*z + 42*y - 80*z - 28;
        },X,dx,dy,dz);
;
  }
  static double deltaf(unsigned i, unsigned j, const Eigen::Vector3d &X, unsigned dx, unsigned dy, unsigned dz) {
    if (i == 0 && j == 0) {
      return autodiff1st(
        [](autodiff::real x, autodiff::real y, autodiff::real z)->autodiff::real {
          return 0;
        },X,dx,dy,dz);
    } else if (i == 0 && j == 1) {
      return autodiff1st(
        [](autodiff::real x, autodiff::real y, autodiff::real z)->autodiff::real {
          return 57*z + 151;
        },X,dx,dy,dz);
    } else if (i == 0 && j == 2) {
      return autodiff1st(
        [](autodiff::real x, autodiff::real y, autodiff::real z)->autodiff::real {
          return 57*y + 2;
        },X,dx,dy,dz);
    } else if (i == 1 && j == 0) {
      return autodiff1st(
        [](autodiff::real x, autodiff::real y, autodiff::real z)->autodiff::real {
          return 57*z + 151;
        },X,dx,dy,dz);
    } else if (i == 1 && j == 1) {
      return autodiff1st(
        [](autodiff::real x, autodiff::real y, autodiff::real z)->autodiff::real {
          return 0;
        },X,dx,dy,dz);
    } else if (i == 1 && j == 2) {
      return autodiff1st(
        [](autodiff::real x, autodiff::real y, autodiff::real z)->autodiff::real {
          return 57*x + 72;
        },X,dx,dy,dz);
    } else if (i == 2 && j == 0) {
      return autodiff1st(
        [](autodiff::real x, autodiff::real y, autodiff::real z)->autodiff::real {
          return 57*y + 2;
        },X,dx,dy,dz);
    } else if (i == 2 && j == 1) {
      return autodiff1st(
        [](autodiff::real x, autodiff::real y, autodiff::real z)->autodiff::real {
          return 57*x + 72;
        },X,dx,dy,dz);
    } else {
      return autodiff1st(
        [](autodiff::real x, autodiff::real y, autodiff::real z)->autodiff::real {
          return 0;
        },X,dx,dy,dz);
    };
  }
};
struct P3C2 {
  static double f(const Eigen::Vector3d &X, unsigned dx, unsigned dy, unsigned dz) {
      return autodiff1st(
        [](autodiff::real x, autodiff::real y, autodiff::real z)->autodiff::real {
          return x*autodiff::detail::sin(y)*autodiff::detail::cos(z);
        },X,dx,dy,dz);
;
  }
  static double deltaf(unsigned i, unsigned j, const Eigen::Vector3d &X, unsigned dx, unsigned dy, unsigned dz) {

    if (i == 0 && j == 0) {
      return autodiff1st(
        [](autodiff::real x, autodiff::real y, autodiff::real z)->autodiff::real {
          return 0;
        },X,dx,dy,dz);
    } else if (i == 0 && j == 1) {
      return autodiff1st(
        [](autodiff::real x, autodiff::real y, autodiff::real z)->autodiff::real {
          return autodiff::detail::cos(y)*autodiff::detail::cos(z);
        },X,dx,dy,dz);
    } else if (i == 0 && j == 2) {
      return autodiff1st(
        [](autodiff::real x, autodiff::real y, autodiff::real z)->autodiff::real {
          return -autodiff::detail::sin(y)*autodiff::detail::sin(z);
        },X,dx,dy,dz);
    } else if (i == 1 && j == 0) {
      return autodiff1st(
        [](autodiff::real x, autodiff::real y, autodiff::real z)->autodiff::real {
          return autodiff::detail::cos(y)*autodiff::detail::cos(z);
        },X,dx,dy,dz);
    } else if (i == 1 && j == 1) {
      return autodiff1st(
        [](autodiff::real x, autodiff::real y, autodiff::real z)->autodiff::real {
          return -x*autodiff::detail::sin(y)*autodiff::detail::cos(z);
        },X,dx,dy,dz);
    } else if (i == 1 && j == 2) {
      return autodiff1st(
        [](autodiff::real x, autodiff::real y, autodiff::real z)->autodiff::real {
          return -x*autodiff::detail::sin(z)*autodiff::detail::cos(y);
        },X,dx,dy,dz);
    } else if (i == 2 && j == 0) {
      return autodiff1st(
        [](autodiff::real x, autodiff::real y, autodiff::real z)->autodiff::real {
          return -autodiff::detail::sin(y)*autodiff::detail::sin(z);
        },X,dx,dy,dz);
    } else if (i == 2 && j == 1) {
      return autodiff1st(
        [](autodiff::real x, autodiff::real y, autodiff::real z)->autodiff::real {
          return -x*autodiff::detail::sin(z)*autodiff::detail::cos(y);
        },X,dx,dy,dz);
    } else {
      return autodiff1st(
        [](autodiff::real x, autodiff::real y, autodiff::real z)->autodiff::real {
          return -x*autodiff::detail::sin(y)*autodiff::detail::cos(z);
        },X,dx,dy,dz);
    };
  }
};

#endif
