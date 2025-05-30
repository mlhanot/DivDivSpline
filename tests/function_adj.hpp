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
          return -(4*x - 1)*(autodiff::detail::pow(y, 2) + 3*y - 1)*(2*autodiff::detail::pow(z, 2) - z + 2) - (autodiff::detail::pow(y, 2) + 3*y - 1)*(2*autodiff::detail::pow(x, 3) + autodiff::detail::pow(x, 2) - x + 1) - 3*(2*autodiff::detail::pow(z, 2) - z + 2)*(autodiff::detail::pow(x, 3) - 2*autodiff::detail::pow(x, 2) - x + 1);
        },X,dx,dy,dz);
    } else if (i == 1) {
      return autodiff1st(
        [](autodiff::real x, autodiff::real y, autodiff::real z)->autodiff::real {
          return (2*y - 1)*(3*autodiff::detail::pow(x, 2) - x + 1)*(3*autodiff::detail::pow(z, 2) - z + 3) - (5*autodiff::detail::pow(x, 2) - x + 1)*(4*autodiff::detail::pow(y, 3) - autodiff::detail::pow(y, 2) + 2*y - 1) - (3*autodiff::detail::pow(z, 2) - z + 3)*(4*autodiff::detail::pow(y, 3) - autodiff::detail::pow(y, 2) + 2*y - 1);
        },X,dx,dy,dz);
    } else {
      return autodiff1st(
        [](autodiff::real x, autodiff::real y, autodiff::real z)->autodiff::real {
          return (4*z - 1)*(2*autodiff::detail::pow(x, 2) - x + 1)*(autodiff::detail::pow(y, 2) + 3*y - 1) - (6*z - 1)*(3*autodiff::detail::pow(x, 2) - x + 1)*(autodiff::detail::pow(y, 2) - y + 1) + (2*autodiff::detail::pow(x, 2) - x + 4)*(autodiff::detail::pow(z, 3) + 2*autodiff::detail::pow(z, 2) + z - 4) + (autodiff::detail::pow(y, 2) - 3*y + 1)*(autodiff::detail::pow(z, 3) + 2*autodiff::detail::pow(z, 2) + z - 3);
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
          return autodiff::detail::sin(x);
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

struct P1A0 {
  static double f(unsigned i, unsigned j, const Eigen::Vector3d &X, unsigned dx, unsigned dy, unsigned dz) {
    if (i == 0 && j == 0) {
      return autodiff1st(
        [](autodiff::real x, autodiff::real y, autodiff::real z)->autodiff::real {
          return -x*y*z*(x - 1)*(y - 1)*(z - 1)*(autodiff::detail::sin(y) + autodiff::detail::cos(x));
        },X,dx,dy,dz);
    } else if (i == 0 && j == 1) {
      return autodiff1st(
        [](autodiff::real x, autodiff::real y, autodiff::real z)->autodiff::real {
          return y*(1 - y)*autodiff::detail::exp(z);
        },X,dx,dy,dz);
    } else if (i == 0 && j == 2) {
      return autodiff1st(
        [](autodiff::real x, autodiff::real y, autodiff::real z)->autodiff::real {
          return -z*(2*x + 1)*(z - 1);
        },X,dx,dy,dz);
    } else if (i == 1 && j == 0) {
      return autodiff1st(
        [](autodiff::real x, autodiff::real y, autodiff::real z)->autodiff::real {
          return x*(1 - x)*autodiff::detail::sin(y);
        },X,dx,dy,dz);
    } else if (i == 1 && j == 1) {
      return autodiff1st(
        [](autodiff::real x, autodiff::real y, autodiff::real z)->autodiff::real {
          return x*y*z*(x - 1)*(y - 1)*(z - 1)*autodiff::detail::cos(x);
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
          return x*y*z*(x - 1)*(y - 1)*(z - 1)*autodiff::detail::sin(y);
        },X,dx,dy,dz);
    };
  }
  static double deltaf(unsigned i, const Eigen::Vector3d &X, unsigned dx, unsigned dy, unsigned dz) {
    if (i == 0) {
      return autodiff1st(
        [](autodiff::real x, autodiff::real y, autodiff::real z)->autodiff::real {
          return -x*y*z*(x - 1)*(y - 1)*(z - 1)*autodiff::detail::sin(x) + x*y*z*(y - 1)*(z - 1)*(autodiff::detail::sin(y) + autodiff::detail::cos(x)) + y*z*(x - 1)*(y - 1)*(z - 1)*(autodiff::detail::sin(y) + autodiff::detail::cos(x)) + y*autodiff::detail::exp(z) + z*(2*x + 1) + (2*x + 1)*(z - 1) + (y - 1)*autodiff::detail::exp(z);
        },X,dx,dy,dz);
    } else if (i == 1) {
      return autodiff1st(
        [](autodiff::real x, autodiff::real y, autodiff::real z)->autodiff::real {
          return -x*y*z*(x - 1)*(z - 1)*autodiff::detail::cos(x) - x*z*(x - 1)*(y - 1)*(z - 1)*autodiff::detail::cos(x) + x*autodiff::detail::sin(y) + (x - 1)*autodiff::detail::sin(y);
        },X,dx,dy,dz);
    } else {
      return autodiff1st(
        [](autodiff::real x, autodiff::real y, autodiff::real z)->autodiff::real {
          return x*y*(1 - 2*z)*(x - 1)*(y - 1)*autodiff::detail::sin(y);
        },X,dx,dy,dz);
    };
  }
};
struct P2A0 {
  static double f(unsigned i, unsigned j, const Eigen::Vector3d &X, unsigned dx, unsigned dy, unsigned dz) {
    if (i == 0 && j == 0) {
      return autodiff1st(
        [](autodiff::real x, autodiff::real y, autodiff::real z)->autodiff::real {
          return y*z*(y - 1)*(z - 1)*(autodiff::detail::sin(y) + autodiff::detail::cos(x));
        },X,dx,dy,dz);
    } else if (i == 0 && j == 1) {
      return autodiff1st(
        [](autodiff::real x, autodiff::real y, autodiff::real z)->autodiff::real {
          return -x*y*z*(x - 1)*(y - 1)*(z - 1)*autodiff::detail::exp(z);
        },X,dx,dy,dz);
    } else if (i == 0 && j == 2) {
      return autodiff1st(
        [](autodiff::real x, autodiff::real y, autodiff::real z)->autodiff::real {
          return -x*y*z*(x - 1)*(2*x + 1)*(y - 1)*(z - 1);
        },X,dx,dy,dz);
    } else if (i == 1 && j == 0) {
      return autodiff1st(
        [](autodiff::real x, autodiff::real y, autodiff::real z)->autodiff::real {
          return -x*y*z*(x - 1)*(y - 1)*(z - 1)*autodiff::detail::exp(z);
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
          return -x*y*z*(x - 1)*(2*x + 1)*(y - 1)*(z - 1);
        },X,dx,dy,dz);
    } else if (i == 2 && j == 1) {
      return autodiff1st(
        [](autodiff::real x, autodiff::real y, autodiff::real z)->autodiff::real {
          return 0;
        },X,dx,dy,dz);
    } else {
      return autodiff1st(
        [](autodiff::real x, autodiff::real y, autodiff::real z)->autodiff::real {
          return x*y*(x - 1)*(y - 1)*autodiff::detail::sin(y);
        },X,dx,dy,dz);
    };
  }
  static double deltaf(unsigned i, unsigned j, const Eigen::Vector3d &X, unsigned dx, unsigned dy, unsigned dz) {
    if (i == 0 && j == 0) {
      return autodiff1st(
        [](autodiff::real x, autodiff::real y, autodiff::real z)->autodiff::real {
          return x*(x - 1)*(-y*z*(2*x + 1)*(z - 1) + y*z*(y - 1)*(z - 1)*autodiff::detail::exp(z) + y*z*(y - 1)*autodiff::detail::exp(z) + y*(y - 1)*(z - 1)*autodiff::detail::exp(z) - z*(2*x + 1)*(y - 1)*(z - 1));
        },X,dx,dy,dz);
    } else if (i == 0 && j == 1) {
      return autodiff1st(
        [](autodiff::real x, autodiff::real y, autodiff::real z)->autodiff::real {
          return y*(y - 1)*(6*autodiff::detail::pow(x, 2)*autodiff::detail::pow(z, 2) - 6*autodiff::detail::pow(x, 2)*z - 2*x*autodiff::detail::pow(z, 2) + 2*x*z - autodiff::detail::pow(z, 2) + 2*z*autodiff::detail::sin(y) + 2*z*autodiff::detail::cos(x) + z - autodiff::detail::sin(y) - autodiff::detail::cos(x));
        },X,dx,dy,dz);
    } else if (i == 0 && j == 2) {
      return autodiff1st(
        [](autodiff::real x, autodiff::real y, autodiff::real z)->autodiff::real {
          return z*(z - 1)*(-x*y*(y - 1)*autodiff::detail::exp(z) - y*(x - 1)*(y - 1)*autodiff::detail::exp(z) - y*(y - 1)*autodiff::detail::cos(y) - y*(autodiff::detail::sin(y) + autodiff::detail::cos(x)) + (1 - y)*(autodiff::detail::sin(y) + autodiff::detail::cos(x)));
        },X,dx,dy,dz);
    } else if (i == 1 && j == 0) {
      return autodiff1st(
        [](autodiff::real x, autodiff::real y, autodiff::real z)->autodiff::real {
          return 0;
        },X,dx,dy,dz);
    } else if (i == 1 && j == 1) {
      return autodiff1st(
        [](autodiff::real x, autodiff::real y, autodiff::real z)->autodiff::real {
          return x*y*(x - 1)*(y - 1)*(-z*(z - 1) - 2*z + 1)*autodiff::detail::exp(z);
        },X,dx,dy,dz);
    } else if (i == 1 && j == 2) {
      return autodiff1st(
        [](autodiff::real x, autodiff::real y, autodiff::real z)->autodiff::real {
          return x*z*(x - 1)*(2*y - 1)*(z - 1)*autodiff::detail::exp(z);
        },X,dx,dy,dz);
    } else if (i == 2 && j == 0) {
      return autodiff1st(
        [](autodiff::real x, autodiff::real y, autodiff::real z)->autodiff::real {
          return x*(x - 1)*(y*(y - 1)*autodiff::detail::cos(y) + y*autodiff::detail::sin(y) + (y - 1)*autodiff::detail::sin(y));
        },X,dx,dy,dz);
    } else if (i == 2 && j == 1) {
      return autodiff1st(
        [](autodiff::real x, autodiff::real y, autodiff::real z)->autodiff::real {
          return y*(y - 1)*(-x*z*(x - 1)*(2*x + 1) - x*(x - 1)*(2*x + 1)*(z - 1) - x*autodiff::detail::sin(y) + (1 - x)*autodiff::detail::sin(y));
        },X,dx,dy,dz);
    } else {
      return autodiff1st(
        [](autodiff::real x, autodiff::real y, autodiff::real z)->autodiff::real {
          return x*z*(x - 1)*(2*x + 1)*(2*y - 1)*(z - 1);
        },X,dx,dy,dz);
    };
  }
};
struct P3A0 {
  static double f(const Eigen::Vector3d &X, unsigned dx, unsigned dy, unsigned dz) {

      return autodiff1st(
        [](autodiff::real x, autodiff::real y, autodiff::real z)->autodiff::real {
          return -autodiff::detail::pow(x, 2)*y*z*(x - 1)*(y - 1)*(z - 1)*autodiff::detail::sin(y)*autodiff::detail::cos(z);
        },X,dx,dy,dz);

  }
  static double deltaf(unsigned i, unsigned j, const Eigen::Vector3d &X, unsigned dx, unsigned dy, unsigned dz) {
    if (i == 0 && j == 0) {
      return autodiff1st(
        [](autodiff::real x, autodiff::real y, autodiff::real z)->autodiff::real {
          return 2*y*z*(1 - 3*x)*(y - 1)*(z - 1)*autodiff::detail::sin(y)*autodiff::detail::cos(z);
        },X,dx,dy,dz);
    } else if (i == 0 && j == 1) {
      return autodiff1st(
        [](autodiff::real x, autodiff::real y, autodiff::real z)->autodiff::real {
          return x*z*(z - 1)*(-x*y*(y - 1)*autodiff::detail::cos(y) - x*y*autodiff::detail::sin(y) - x*(y - 1)*autodiff::detail::sin(y) - 2*y*(x - 1)*(y - 1)*autodiff::detail::cos(y) - 2*y*(x - 1)*autodiff::detail::sin(y) - 2*(x - 1)*(y - 1)*autodiff::detail::sin(y))*autodiff::detail::cos(z);
        },X,dx,dy,dz);
    } else if (i == 0 && j == 2) {
      return autodiff1st(
        [](autodiff::real x, autodiff::real y, autodiff::real z)->autodiff::real {
          return x*y*(y - 1)*(x*z*(z - 1)*autodiff::detail::sin(z) - x*z*autodiff::detail::cos(z) - x*(z - 1)*autodiff::detail::cos(z) + 2*z*(x - 1)*(z - 1)*autodiff::detail::sin(z) - 2*z*(x - 1)*autodiff::detail::cos(z) - 2*(x - 1)*(z - 1)*autodiff::detail::cos(z))*autodiff::detail::sin(y);
        },X,dx,dy,dz);
    } else if (i == 1 && j == 0) {
      return autodiff1st(
        [](autodiff::real x, autodiff::real y, autodiff::real z)->autodiff::real {
          return x*z*(z - 1)*(-x*y*(y - 1)*autodiff::detail::cos(y) - x*y*autodiff::detail::sin(y) - x*(y - 1)*autodiff::detail::sin(y) - 2*y*(x - 1)*(y - 1)*autodiff::detail::cos(y) - 2*y*(x - 1)*autodiff::detail::sin(y) - 2*(x - 1)*(y - 1)*autodiff::detail::sin(y))*autodiff::detail::cos(z);
        },X,dx,dy,dz);
    } else if (i == 1 && j == 1) {
      return autodiff1st(
        [](autodiff::real x, autodiff::real y, autodiff::real z)->autodiff::real {
          return autodiff::detail::pow(x, 2)*z*(x - 1)*(z - 1)*(y*(y - 1)*autodiff::detail::sin(y) - 2*y*autodiff::detail::cos(y) + (2 - 2*y)*autodiff::detail::cos(y) - 2*autodiff::detail::sin(y))*autodiff::detail::cos(z);
        },X,dx,dy,dz);
    } else if (i == 1 && j == 2) {
      return autodiff1st(
        [](autodiff::real x, autodiff::real y, autodiff::real z)->autodiff::real {
          return autodiff::detail::pow(x, 2)*(x - 1)*(y*z*(y - 1)*(z - 1)*autodiff::detail::sin(z)*autodiff::detail::cos(y) - y*z*(y - 1)*autodiff::detail::cos(y)*autodiff::detail::cos(z) + y*z*(z - 1)*autodiff::detail::sin(y)*autodiff::detail::sin(z) - y*z*autodiff::detail::sin(y)*autodiff::detail::cos(z) - y*(y - 1)*(z - 1)*autodiff::detail::cos(y)*autodiff::detail::cos(z) - y*(z - 1)*autodiff::detail::sin(y)*autodiff::detail::cos(z) + z*(y - 1)*(z - 1)*autodiff::detail::sin(y)*autodiff::detail::sin(z) - z*(y - 1)*autodiff::detail::sin(y)*autodiff::detail::cos(z) - (y - 1)*(z - 1)*autodiff::detail::sin(y)*autodiff::detail::cos(z));
        },X,dx,dy,dz);
    } else if (i == 2 && j == 0) {
      return autodiff1st(
        [](autodiff::real x, autodiff::real y, autodiff::real z)->autodiff::real {
          return x*y*(y - 1)*(x*z*(z - 1)*autodiff::detail::sin(z) - x*z*autodiff::detail::cos(z) - x*(z - 1)*autodiff::detail::cos(z) + 2*z*(x - 1)*(z - 1)*autodiff::detail::sin(z) - 2*z*(x - 1)*autodiff::detail::cos(z) - 2*(x - 1)*(z - 1)*autodiff::detail::cos(z))*autodiff::detail::sin(y);
        },X,dx,dy,dz);
    } else if (i == 2 && j == 1) {
      return autodiff1st(
        [](autodiff::real x, autodiff::real y, autodiff::real z)->autodiff::real {
          return autodiff::detail::pow(x, 2)*(x - 1)*(y*z*(y - 1)*(z - 1)*autodiff::detail::sin(z)*autodiff::detail::cos(y) - y*z*(y - 1)*autodiff::detail::cos(y)*autodiff::detail::cos(z) + y*z*(z - 1)*autodiff::detail::sin(y)*autodiff::detail::sin(z) - y*z*autodiff::detail::sin(y)*autodiff::detail::cos(z) - y*(y - 1)*(z - 1)*autodiff::detail::cos(y)*autodiff::detail::cos(z) - y*(z - 1)*autodiff::detail::sin(y)*autodiff::detail::cos(z) + z*(y - 1)*(z - 1)*autodiff::detail::sin(y)*autodiff::detail::sin(z) - z*(y - 1)*autodiff::detail::sin(y)*autodiff::detail::cos(z) - (y - 1)*(z - 1)*autodiff::detail::sin(y)*autodiff::detail::cos(z));
        },X,dx,dy,dz);
    } else {
      return autodiff1st(
        [](autodiff::real x, autodiff::real y, autodiff::real z)->autodiff::real {
          return autodiff::detail::pow(x, 2)*y*(x - 1)*(y - 1)*(z*(z - 1)*autodiff::detail::cos(z) + 2*z*autodiff::detail::sin(z) + (2*z - 2)*autodiff::detail::sin(z) - 2*autodiff::detail::cos(z))*autodiff::detail::sin(y);
        },X,dx,dy,dz);
    };
  }
};

struct P1A1 {
  static double f(unsigned i, unsigned j, const Eigen::Vector3d &X, unsigned dx, unsigned dy, unsigned dz) {
    if (i == 0 && j == 0) {
      return autodiff1st(
        [](autodiff::real x, autodiff::real y, autodiff::real z)->autodiff::real {
          return 0;
        },X,dx,dy,dz);
    } else if (i == 0 && j == 1) {
      return autodiff1st(
        [](autodiff::real x, autodiff::real y, autodiff::real z)->autodiff::real {
          return autodiff::detail::pow(x, 2)*autodiff::detail::pow(y, 2)*autodiff::detail::pow(z, 2)*autodiff::detail::pow(x - 1, 2)*autodiff::detail::pow(y - 1, 2)*autodiff::detail::pow(z - 1, 2);
        },X,dx,dy,dz);
    } else if (i == 0 && j == 2) {
      return autodiff1st(
        [](autodiff::real x, autodiff::real y, autodiff::real z)->autodiff::real {
          return 2*autodiff::detail::pow(x, 2)*autodiff::detail::pow(y, 2)*autodiff::detail::pow(z, 2)*autodiff::detail::pow(x - 1, 2)*autodiff::detail::pow(y - 1, 2)*autodiff::detail::pow(z - 1, 2);
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
          return 2*autodiff::detail::pow(x, 2)*y*z*autodiff::detail::pow(x - 1, 2)*(y*z*(1 - y)*autodiff::detail::pow(z - 1, 2) + 2*y*z*(1 - z)*autodiff::detail::pow(y - 1, 2) - 2*y*autodiff::detail::pow(y - 1, 2)*autodiff::detail::pow(z - 1, 2) - z*autodiff::detail::pow(y - 1, 2)*autodiff::detail::pow(z - 1, 2));
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
struct P2A1 {
  static double f(unsigned i, unsigned j, const Eigen::Vector3d &X, unsigned dx, unsigned dy, unsigned dz) {
    if (i == 0 && j == 0) {
      return autodiff1st(
        [](autodiff::real x, autodiff::real y, autodiff::real z)->autodiff::real {
          return 0;
        },X,dx,dy,dz);
    } else if (i == 0 && j == 1) {
      return autodiff1st(
        [](autodiff::real x, autodiff::real y, autodiff::real z)->autodiff::real {
          return autodiff::detail::pow(x, 2)*autodiff::detail::pow(y, 2)*autodiff::detail::pow(z, 2)*autodiff::detail::pow(x - 1, 2)*autodiff::detail::pow(y - 1, 2)*autodiff::detail::pow(z - 1, 2);
        },X,dx,dy,dz);
    } else if (i == 0 && j == 2) {
      return autodiff1st(
        [](autodiff::real x, autodiff::real y, autodiff::real z)->autodiff::real {
          return 2*autodiff::detail::pow(x, 2)*autodiff::detail::pow(y, 2)*autodiff::detail::pow(z, 2)*autodiff::detail::pow(x - 1, 2)*autodiff::detail::pow(y - 1, 2)*autodiff::detail::pow(z - 1, 2);
        },X,dx,dy,dz);
    } else if (i == 1 && j == 0) {
      return autodiff1st(
        [](autodiff::real x, autodiff::real y, autodiff::real z)->autodiff::real {
          return autodiff::detail::pow(x, 2)*autodiff::detail::pow(y, 2)*autodiff::detail::pow(z, 2)*autodiff::detail::pow(x - 1, 2)*autodiff::detail::pow(y - 1, 2)*autodiff::detail::pow(z - 1, 2);
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
          return 2*autodiff::detail::pow(x, 2)*autodiff::detail::pow(y, 2)*autodiff::detail::pow(z, 2)*autodiff::detail::pow(x - 1, 2)*autodiff::detail::pow(y - 1, 2)*autodiff::detail::pow(z - 1, 2);
        },X,dx,dy,dz);
    } else if (i == 2 && j == 1) {
      return autodiff1st(
        [](autodiff::real x, autodiff::real y, autodiff::real z)->autodiff::real {
          return 0;
        },X,dx,dy,dz);
    } else {
      return autodiff1st(
        [](autodiff::real x, autodiff::real y, autodiff::real z)->autodiff::real {
          return 3*autodiff::detail::pow(x, 2)*autodiff::detail::pow(y, 2)*autodiff::detail::pow(z, 2)*autodiff::detail::pow(x - 1, 2)*autodiff::detail::pow(y - 1, 2)*autodiff::detail::pow(z - 1, 2);
        },X,dx,dy,dz);
    };
  }
  static double deltaf(unsigned i, unsigned j, const Eigen::Vector3d &X, unsigned dx, unsigned dy, unsigned dz) {
    if (i == 0 && j == 0) {
      return autodiff1st(
        [](autodiff::real x, autodiff::real y, autodiff::real z)->autodiff::real {
          return 2*autodiff::detail::pow(x, 2)*y*z*autodiff::detail::pow(x - 1, 2)*(y - 1)*(y*z*(1 - z)*(y - 1) + 2*y*z*autodiff::detail::pow(z - 1, 2) - y*(y - 1)*autodiff::detail::pow(z - 1, 2) + 2*z*(y - 1)*autodiff::detail::pow(z - 1, 2));
        },X,dx,dy,dz);
    } else if (i == 0 && j == 1) {
      return autodiff1st(
        [](autodiff::real x, autodiff::real y, autodiff::real z)->autodiff::real {
          return 4*x*autodiff::detail::pow(y, 2)*autodiff::detail::pow(z, 2)*autodiff::detail::pow(y - 1, 2)*autodiff::detail::pow(z - 1, 2)*(x*(1 - x) - autodiff::detail::pow(x - 1, 2));
        },X,dx,dy,dz);
    } else if (i == 0 && j == 2) {
      return autodiff1st(
        [](autodiff::real x, autodiff::real y, autodiff::real z)->autodiff::real {
          return 2*x*autodiff::detail::pow(y, 2)*autodiff::detail::pow(z, 2)*(x - 1)*(2*x - 1)*autodiff::detail::pow(y - 1, 2)*autodiff::detail::pow(z - 1, 2);
        },X,dx,dy,dz);
    } else if (i == 1 && j == 0) {
      return autodiff1st(
        [](autodiff::real x, autodiff::real y, autodiff::real z)->autodiff::real {
          return 0;
        },X,dx,dy,dz);
    } else if (i == 1 && j == 1) {
      return autodiff1st(
        [](autodiff::real x, autodiff::real y, autodiff::real z)->autodiff::real {
          return 2*autodiff::detail::pow(x, 2)*autodiff::detail::pow(y, 2)*z*autodiff::detail::pow(x - 1, 2)*autodiff::detail::pow(y - 1, 2)*(z - 1)*(2*z - 1);
        },X,dx,dy,dz);
    } else if (i == 1 && j == 2) {
      return autodiff1st(
        [](autodiff::real x, autodiff::real y, autodiff::real z)->autodiff::real {
          return 2*autodiff::detail::pow(x, 2)*y*autodiff::detail::pow(z, 2)*autodiff::detail::pow(x - 1, 2)*autodiff::detail::pow(z - 1, 2)*(y*(1 - y) - autodiff::detail::pow(y - 1, 2));
        },X,dx,dy,dz);
    } else if (i == 2 && j == 0) {
      return autodiff1st(
        [](autodiff::real x, autodiff::real y, autodiff::real z)->autodiff::real {
          return 6*autodiff::detail::pow(x, 2)*y*autodiff::detail::pow(z, 2)*autodiff::detail::pow(x - 1, 2)*(y - 1)*(2*y - 1)*autodiff::detail::pow(z - 1, 2);
        },X,dx,dy,dz);
    } else if (i == 2 && j == 1) {
      return autodiff1st(
        [](autodiff::real x, autodiff::real y, autodiff::real z)->autodiff::real {
          return 2*x*autodiff::detail::pow(y, 2)*z*autodiff::detail::pow(y - 1, 2)*(z - 1)*(3*x*z*(1 - x)*(z - 1) + 2*x*z*autodiff::detail::pow(x - 1, 2) + 2*x*autodiff::detail::pow(x - 1, 2)*(z - 1) - 3*z*autodiff::detail::pow(x - 1, 2)*(z - 1));
        },X,dx,dy,dz);
    } else {
      return autodiff1st(
        [](autodiff::real x, autodiff::real y, autodiff::real z)->autodiff::real {
          return 4*autodiff::detail::pow(x, 2)*y*autodiff::detail::pow(z, 2)*autodiff::detail::pow(x - 1, 2)*autodiff::detail::pow(z - 1, 2)*(y*(1 - y) - autodiff::detail::pow(y - 1, 2));
        },X,dx,dy,dz);
    };
  }
};
struct P3A1 {
  static double f(const Eigen::Vector3d &X, unsigned dx, unsigned dy, unsigned dz) {
      return autodiff1st(
        [](autodiff::real x, autodiff::real y, autodiff::real z)->autodiff::real {
          return 3*autodiff::detail::pow(x, 2)*autodiff::detail::pow(y, 2)*autodiff::detail::pow(z, 2)*autodiff::detail::pow(x - 1, 2)*autodiff::detail::pow(y - 1, 2)*autodiff::detail::pow(z - 1, 2);
        },X,dx,dy,dz);

  }
  static double deltaf(unsigned i, unsigned j, const Eigen::Vector3d &X, unsigned dx, unsigned dy, unsigned dz) {

    if (i == 0 && j == 0) {
      return autodiff1st(
        [](autodiff::real x, autodiff::real y, autodiff::real z)->autodiff::real {
          return 6*autodiff::detail::pow(y, 2)*autodiff::detail::pow(z, 2)*autodiff::detail::pow(y - 1, 2)*autodiff::detail::pow(z - 1, 2)*(autodiff::detail::pow(x, 2) + 4*x*(x - 1) + autodiff::detail::pow(x - 1, 2));
        },X,dx,dy,dz);
    } else if (i == 0 && j == 1) {
      return autodiff1st(
        [](autodiff::real x, autodiff::real y, autodiff::real z)->autodiff::real {
          return 12*x*y*autodiff::detail::pow(z, 2)*(x - 1)*(y - 1)*autodiff::detail::pow(z - 1, 2)*(x*y + x*(y - 1) + y*(x - 1) + (x - 1)*(y - 1));
        },X,dx,dy,dz);
    } else if (i == 0 && j == 2) {
      return autodiff1st(
        [](autodiff::real x, autodiff::real y, autodiff::real z)->autodiff::real {
          return 12*x*autodiff::detail::pow(y, 2)*z*(x - 1)*autodiff::detail::pow(y - 1, 2)*(z - 1)*(x*z + x*(z - 1) + z*(x - 1) + (x - 1)*(z - 1));
        },X,dx,dy,dz);
    } else if (i == 1 && j == 0) {
      return autodiff1st(
        [](autodiff::real x, autodiff::real y, autodiff::real z)->autodiff::real {
          return 12*x*y*autodiff::detail::pow(z, 2)*(x - 1)*(y - 1)*autodiff::detail::pow(z - 1, 2)*(x*y + x*(y - 1) + y*(x - 1) + (x - 1)*(y - 1));
        },X,dx,dy,dz);
    } else if (i == 1 && j == 1) {
      return autodiff1st(
        [](autodiff::real x, autodiff::real y, autodiff::real z)->autodiff::real {
          return 6*autodiff::detail::pow(x, 2)*autodiff::detail::pow(z, 2)*autodiff::detail::pow(x - 1, 2)*autodiff::detail::pow(z - 1, 2)*(autodiff::detail::pow(y, 2) + 4*y*(y - 1) + autodiff::detail::pow(y - 1, 2));
        },X,dx,dy,dz);
    } else if (i == 1 && j == 2) {
      return autodiff1st(
        [](autodiff::real x, autodiff::real y, autodiff::real z)->autodiff::real {
          return 12*autodiff::detail::pow(x, 2)*y*z*autodiff::detail::pow(x - 1, 2)*(y - 1)*(z - 1)*(y*z + y*(z - 1) + z*(y - 1) + (y - 1)*(z - 1));
        },X,dx,dy,dz);
    } else if (i == 2 && j == 0) {
      return autodiff1st(
        [](autodiff::real x, autodiff::real y, autodiff::real z)->autodiff::real {
          return 12*x*autodiff::detail::pow(y, 2)*z*(x - 1)*autodiff::detail::pow(y - 1, 2)*(z - 1)*(x*z + x*(z - 1) + z*(x - 1) + (x - 1)*(z - 1));
        },X,dx,dy,dz);
    } else if (i == 2 && j == 1) {
      return autodiff1st(
        [](autodiff::real x, autodiff::real y, autodiff::real z)->autodiff::real {
          return 12*autodiff::detail::pow(x, 2)*y*z*autodiff::detail::pow(x - 1, 2)*(y - 1)*(z - 1)*(y*z + y*(z - 1) + z*(y - 1) + (y - 1)*(z - 1));
        },X,dx,dy,dz);
    } else {
      return autodiff1st(
        [](autodiff::real x, autodiff::real y, autodiff::real z)->autodiff::real {
          return 6*autodiff::detail::pow(x, 2)*autodiff::detail::pow(y, 2)*autodiff::detail::pow(x - 1, 2)*autodiff::detail::pow(y - 1, 2)*(autodiff::detail::pow(z, 2) + 4*z*(z - 1) + autodiff::detail::pow(z - 1, 2));
        },X,dx,dy,dz);
    };
  }
};

#endif
