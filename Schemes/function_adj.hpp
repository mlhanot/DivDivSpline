#ifndef FUNCTION_HPP
#define FUNCTION_HPP

#include <Eigen/Dense>
#include "../tests/autodiff/forward/real.hpp"

/// Wrapper for first order derivative
template<typename F>
double autodiff1st(F ADf, const Eigen::Vector3d &X, double _t, unsigned dx, unsigned dy, unsigned dz) {
  assert(dx + dy + dz < 2 && "only first order derivative implemented");
  autodiff::real x = X(0), y = X(1), z = X(2), t = _t;
  if (dx > 0) {
    return autodiff::derivative(ADf,autodiff::wrt(x), autodiff::at(x,y,z,t));
  } else if (dy > 0) {
    return autodiff::derivative(ADf,autodiff::wrt(y), autodiff::at(x,y,z,t));
  } else if (dz > 0) {
    return autodiff::derivative(ADf,autodiff::wrt(z), autodiff::at(x,y,z,t));
  } else {
    return ADf(x,y,z,t).val();
  }
};

struct Sol0 {
  double t;
  static constexpr double c = std::acos(-1.), lambda_1 = 2., lambda_2 = 1.;
  double f1(unsigned i, unsigned j, const Eigen::Vector3d &X, unsigned dx, unsigned dy, unsigned dz) {
    if (i == 0 && j == 0) {
      return autodiff1st(
        [](autodiff::real x, autodiff::real y, autodiff::real z, autodiff::real t)->autodiff::real {
          return 0;
        },X,t,dx,dy,dz);
    } else if (i == 0 && j == 1) {
      return autodiff1st(
        [](autodiff::real x, autodiff::real y, autodiff::real z, autodiff::real t)->autodiff::real {
          return lambda_1*autodiff::detail::cos(c*(t - z));
        },X,t,dx,dy,dz);
    } else if (i == 0 && j == 2) {
      return autodiff1st(
        [](autodiff::real x, autodiff::real y, autodiff::real z, autodiff::real t)->autodiff::real {
          return 0;
        },X,t,dx,dy,dz);
    } else if (i == 1 && j == 0) {
      return autodiff1st(
        [](autodiff::real x, autodiff::real y, autodiff::real z, autodiff::real t)->autodiff::real {
          return -lambda_2*autodiff::detail::cos(c*(t - z));
        },X,t,dx,dy,dz);
    } else if (i == 1 && j == 1) {
      return autodiff1st(
        [](autodiff::real x, autodiff::real y, autodiff::real z, autodiff::real t)->autodiff::real {
          return 0;
        },X,t,dx,dy,dz);
    } else if (i == 1 && j == 2) {
      return autodiff1st(
        [](autodiff::real x, autodiff::real y, autodiff::real z, autodiff::real t)->autodiff::real {
          return 0;
        },X,t,dx,dy,dz);
    } else if (i == 2 && j == 0) {
      return autodiff1st(
        [](autodiff::real x, autodiff::real y, autodiff::real z, autodiff::real t)->autodiff::real {
          return 0;
        },X,t,dx,dy,dz);
    } else if (i == 2 && j == 1) {
      return autodiff1st(
        [](autodiff::real x, autodiff::real y, autodiff::real z, autodiff::real t)->autodiff::real {
          return 0;
        },X,t,dx,dy,dz);
    } else {
      return autodiff1st(
        [](autodiff::real x, autodiff::real y, autodiff::real z, autodiff::real t)->autodiff::real {
          return 0;
        },X,t,dx,dy,dz);
    };
  }
  double df1(unsigned i, unsigned j, const Eigen::Vector3d &X, unsigned dx, unsigned dy, unsigned dz) {
    if (i == 0 && j == 0) {
      return autodiff1st(
        [](autodiff::real x, autodiff::real y, autodiff::real z, autodiff::real t)->autodiff::real {
          return -c*lambda_1*autodiff::detail::sin(c*(t - z));
        },X,t,dx,dy,dz);
    } else if (i == 0 && j == 1) {
      return autodiff1st(
        [](autodiff::real x, autodiff::real y, autodiff::real z, autodiff::real t)->autodiff::real {
          return 0;
        },X,t,dx,dy,dz);
    } else if (i == 0 && j == 2) {
      return autodiff1st(
        [](autodiff::real x, autodiff::real y, autodiff::real z, autodiff::real t)->autodiff::real {
          return 0;
        },X,t,dx,dy,dz);
    } else if (i == 1 && j == 0) {
      return autodiff1st(
        [](autodiff::real x, autodiff::real y, autodiff::real z, autodiff::real t)->autodiff::real {
          return 0;
        },X,t,dx,dy,dz);
    } else if (i == 1 && j == 1) {
      return autodiff1st(
        [](autodiff::real x, autodiff::real y, autodiff::real z, autodiff::real t)->autodiff::real {
          return -c*lambda_2*autodiff::detail::sin(c*(t - z));
        },X,t,dx,dy,dz);
    } else if (i == 1 && j == 2) {
      return autodiff1st(
        [](autodiff::real x, autodiff::real y, autodiff::real z, autodiff::real t)->autodiff::real {
          return 0;
        },X,t,dx,dy,dz);
    } else if (i == 2 && j == 0) {
      return autodiff1st(
        [](autodiff::real x, autodiff::real y, autodiff::real z, autodiff::real t)->autodiff::real {
          return 0;
        },X,t,dx,dy,dz);
    } else if (i == 2 && j == 1) {
      return autodiff1st(
        [](autodiff::real x, autodiff::real y, autodiff::real z, autodiff::real t)->autodiff::real {
          return 0;
        },X,t,dx,dy,dz);
    } else {
      return autodiff1st(
        [](autodiff::real x, autodiff::real y, autodiff::real z, autodiff::real t)->autodiff::real {
          return 0;
        },X,t,dx,dy,dz);
    };
  }
  double f2(unsigned i, unsigned j, const Eigen::Vector3d &X, unsigned dx, unsigned dy, unsigned dz) {
    if (i == 0 && j == 0) {
      return autodiff1st(
        [](autodiff::real x, autodiff::real y, autodiff::real z, autodiff::real t)->autodiff::real {
          return lambda_1*autodiff::detail::cos(c*(t - z));
        },X,t,dx,dy,dz);
    } else if (i == 0 && j == 1) {
      return autodiff1st(
        [](autodiff::real x, autodiff::real y, autodiff::real z, autodiff::real t)->autodiff::real {
          return 0;
        },X,t,dx,dy,dz);
    } else if (i == 0 && j == 2) {
      return autodiff1st(
        [](autodiff::real x, autodiff::real y, autodiff::real z, autodiff::real t)->autodiff::real {
          return 0;
        },X,t,dx,dy,dz);
    } else if (i == 1 && j == 0) {
      return autodiff1st(
        [](autodiff::real x, autodiff::real y, autodiff::real z, autodiff::real t)->autodiff::real {
          return 0;
        },X,t,dx,dy,dz);
    } else if (i == 1 && j == 1) {
      return autodiff1st(
        [](autodiff::real x, autodiff::real y, autodiff::real z, autodiff::real t)->autodiff::real {
          return lambda_2*autodiff::detail::cos(c*(t - z));
        },X,t,dx,dy,dz);
    } else if (i == 1 && j == 2) {
      return autodiff1st(
        [](autodiff::real x, autodiff::real y, autodiff::real z, autodiff::real t)->autodiff::real {
          return 0;
        },X,t,dx,dy,dz);
    } else if (i == 2 && j == 0) {
      return autodiff1st(
        [](autodiff::real x, autodiff::real y, autodiff::real z, autodiff::real t)->autodiff::real {
          return 0;
        },X,t,dx,dy,dz);
    } else if (i == 2 && j == 1) {
      return autodiff1st(
        [](autodiff::real x, autodiff::real y, autodiff::real z, autodiff::real t)->autodiff::real {
          return 0;
        },X,t,dx,dy,dz);
    } else {
      return autodiff1st(
        [](autodiff::real x, autodiff::real y, autodiff::real z, autodiff::real t)->autodiff::real {
          return 0;
        },X,t,dx,dy,dz);
    };
  }
  double df2(const Eigen::Vector3d &X, unsigned dx, unsigned dy, unsigned dz) {
      return autodiff1st(
        [](autodiff::real x, autodiff::real y, autodiff::real z, autodiff::real t)->autodiff::real {
          return 0;
        },X,t,dx,dy,dz);

  }
};
struct Sol1 {
  double t;
  static constexpr double c = std::acos(-1.), lambda_1 = 2., lambda_2 = 1.;
  double f1(unsigned i, unsigned j, const Eigen::Vector3d &X, unsigned dx, unsigned dy, unsigned dz) {
    if (i == 0 && j == 0) {
      return autodiff1st(
        [](autodiff::real x, autodiff::real y, autodiff::real z, autodiff::real t)->autodiff::real {
          return 0;
        },X,t,dx,dy,dz);
    } else if (i == 0 && j == 1) {
      return autodiff1st(
        [](autodiff::real x, autodiff::real y, autodiff::real z, autodiff::real t)->autodiff::real {
          return 0;
        },X,t,dx,dy,dz);
    } else if (i == 0 && j == 2) {
      return autodiff1st(
        [](autodiff::real x, autodiff::real y, autodiff::real z, autodiff::real t)->autodiff::real {
          return -lambda_1*autodiff::detail::cos(c*(-t + x + y));
        },X,t,dx,dy,dz);
    } else if (i == 1 && j == 0) {
      return autodiff1st(
        [](autodiff::real x, autodiff::real y, autodiff::real z, autodiff::real t)->autodiff::real {
          return 0;
        },X,t,dx,dy,dz);
    } else if (i == 1 && j == 1) {
      return autodiff1st(
        [](autodiff::real x, autodiff::real y, autodiff::real z, autodiff::real t)->autodiff::real {
          return 0;
        },X,t,dx,dy,dz);
    } else if (i == 1 && j == 2) {
      return autodiff1st(
        [](autodiff::real x, autodiff::real y, autodiff::real z, autodiff::real t)->autodiff::real {
          return -lambda_1*autodiff::detail::cos(c*(-t + x + y));
        },X,t,dx,dy,dz);
    } else if (i == 2 && j == 0) {
      return autodiff1st(
        [](autodiff::real x, autodiff::real y, autodiff::real z, autodiff::real t)->autodiff::real {
          return 0;
        },X,t,dx,dy,dz);
    } else if (i == 2 && j == 1) {
      return autodiff1st(
        [](autodiff::real x, autodiff::real y, autodiff::real z, autodiff::real t)->autodiff::real {
          return 0;
        },X,t,dx,dy,dz);
    } else {
      return autodiff1st(
        [](autodiff::real x, autodiff::real y, autodiff::real z, autodiff::real t)->autodiff::real {
          return 0;
        },X,t,dx,dy,dz);
    };
  }
  double df1(unsigned i, unsigned j, const Eigen::Vector3d &X, unsigned dx, unsigned dy, unsigned dz) {

    if (i == 0 && j == 0) {
      return autodiff1st(
        [](autodiff::real x, autodiff::real y, autodiff::real z, autodiff::real t)->autodiff::real {
          return c*lambda_1*autodiff::detail::sin(c*(-t + x + y));
        },X,t,dx,dy,dz);
    } else if (i == 0 && j == 1) {
      return autodiff1st(
        [](autodiff::real x, autodiff::real y, autodiff::real z, autodiff::real t)->autodiff::real {
          return 0;
        },X,t,dx,dy,dz);
    } else if (i == 0 && j == 2) {
      return autodiff1st(
        [](autodiff::real x, autodiff::real y, autodiff::real z, autodiff::real t)->autodiff::real {
          return 0;
        },X,t,dx,dy,dz);
    } else if (i == 1 && j == 0) {
      return autodiff1st(
        [](autodiff::real x, autodiff::real y, autodiff::real z, autodiff::real t)->autodiff::real {
          return 0;
        },X,t,dx,dy,dz);
    } else if (i == 1 && j == 1) {
      return autodiff1st(
        [](autodiff::real x, autodiff::real y, autodiff::real z, autodiff::real t)->autodiff::real {
          return -c*lambda_1*autodiff::detail::sin(c*(-t + x + y));
        },X,t,dx,dy,dz);
    } else if (i == 1 && j == 2) {
      return autodiff1st(
        [](autodiff::real x, autodiff::real y, autodiff::real z, autodiff::real t)->autodiff::real {
          return 0;
        },X,t,dx,dy,dz);
    } else if (i == 2 && j == 0) {
      return autodiff1st(
        [](autodiff::real x, autodiff::real y, autodiff::real z, autodiff::real t)->autodiff::real {
          return 0;
        },X,t,dx,dy,dz);
    } else if (i == 2 && j == 1) {
      return autodiff1st(
        [](autodiff::real x, autodiff::real y, autodiff::real z, autodiff::real t)->autodiff::real {
          return 0;
        },X,t,dx,dy,dz);
    } else {
      return autodiff1st(
        [](autodiff::real x, autodiff::real y, autodiff::real z, autodiff::real t)->autodiff::real {
          return 0;
        },X,t,dx,dy,dz);
    };
  }
  double f2(unsigned i, unsigned j, const Eigen::Vector3d &X, unsigned dx, unsigned dy, unsigned dz) {
    if (i == 0 && j == 0) {
      return autodiff1st(
        [](autodiff::real x, autodiff::real y, autodiff::real z, autodiff::real t)->autodiff::real {
          return lambda_1*autodiff::detail::cos(c*(-t + x + y));
        },X,t,dx,dy,dz);
    } else if (i == 0 && j == 1) {
      return autodiff1st(
        [](autodiff::real x, autodiff::real y, autodiff::real z, autodiff::real t)->autodiff::real {
          return 0;
        },X,t,dx,dy,dz);
    } else if (i == 0 && j == 2) {
      return autodiff1st(
        [](autodiff::real x, autodiff::real y, autodiff::real z, autodiff::real t)->autodiff::real {
          return 0;
        },X,t,dx,dy,dz);
    } else if (i == 1 && j == 0) {
      return autodiff1st(
        [](autodiff::real x, autodiff::real y, autodiff::real z, autodiff::real t)->autodiff::real {
          return 0;
        },X,t,dx,dy,dz);
    } else if (i == 1 && j == 1) {
      return autodiff1st(
        [](autodiff::real x, autodiff::real y, autodiff::real z, autodiff::real t)->autodiff::real {
          return -lambda_1*autodiff::detail::cos(c*(-t + x + y));
        },X,t,dx,dy,dz);
    } else if (i == 1 && j == 2) {
      return autodiff1st(
        [](autodiff::real x, autodiff::real y, autodiff::real z, autodiff::real t)->autodiff::real {
          return 0;
        },X,t,dx,dy,dz);
    } else if (i == 2 && j == 0) {
      return autodiff1st(
        [](autodiff::real x, autodiff::real y, autodiff::real z, autodiff::real t)->autodiff::real {
          return 0;
        },X,t,dx,dy,dz);
    } else if (i == 2 && j == 1) {
      return autodiff1st(
        [](autodiff::real x, autodiff::real y, autodiff::real z, autodiff::real t)->autodiff::real {
          return 0;
        },X,t,dx,dy,dz);
    } else {
      return autodiff1st(
        [](autodiff::real x, autodiff::real y, autodiff::real z, autodiff::real t)->autodiff::real {
          return 0;
        },X,t,dx,dy,dz);
    };
  }
  double df2(const Eigen::Vector3d &X, unsigned dx, unsigned dy, unsigned dz) {
      return autodiff1st(
        [](autodiff::real x, autodiff::real y, autodiff::real z, autodiff::real t)->autodiff::real {
          return 0;
        },X,t,dx,dy,dz);

  }
};

#endif
