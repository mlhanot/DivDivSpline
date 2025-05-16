#ifndef SPLINE_HPP
#define SPLINE_HPP

#include <cassert>
#include <cmath>
#include <vector>
#include <array>

#include <Eigen/Dense>

#include "legendregauss.hpp"

template<int,int> struct SplineDofs;
template<int,int> class Spline;

/// Class for a 1-dimensionnal spline
/**
  Implement basic operations on some splines.

  phi(int i, double x) returns the evaluation of the i-th shape function at x (in [0,h]).
  IV(int i, int r) returns the i-th interpolator associated with vertices.
  IE(int i, int r) returns the i-th interpolator associated with the edge.

  The shape functions are the dual basis of the interpolator, and are ordered as [IV(v_0), IV(v_1), IE].

  \tparam _d Polynomial degree
  \tparam _r Regularity
  */
template<int _d, int _r>
class SplineBase {
  public:
    SplineBase(double h) : _h(h) {
      LegendreGauss quad(2*dim);
      _mass.setZero();
      for (size_t iqn = 0; iqn < quad.npts(); ++iqn) {
        for (size_t i = 0; i < dim; ++i) {
          const double phii = static_cast<Spline<_d,_r>*>(this)->phi(i,_h*quad.tq(iqn));
          _mass(i,i) += _h*quad.wq(iqn)*phii*phii;
          for (size_t j = i+1; j < dim; ++j) {
            _mass(i,j) += _h*quad.wq(iqn)*phii*static_cast<Spline<_d,_r>*>(this)->phi(j,_h*quad.tq(iqn));
          }
        }
      }
      for (size_t i = 0; i < dim; ++i) {
        for (size_t j = i+1; j < dim; ++j) {
          _mass(j,i) = _mass(i,j);
        }
      }
    }
    static constexpr int degree = _d, regularity = _r, nextD = _d - 1, nextReg = _r - 1;
    static constexpr int dim = _d+1;
    static constexpr int dofV = SplineDofs<_d,_r>::dofV, dofE = SplineDofs<_d,_r>::dofE;
    static constexpr std::array<int,2> dof = {dofV,dofE};
    /// Return the mass matrix of the polynomial basis
    const Eigen::Matrix<double,dim,dim>& mass() const {return _mass;}
    /// The interpolator data is [derivative order][(w,x) : quadrature points]
    using InterpolatorType = std::vector<std::vector<std::array<double,2>>>;
  protected:
    const double _h;
    std::array<InterpolatorType,dofV> _IV;
    std::array<InterpolatorType,dofE> _IE;
    Eigen::Matrix<double,dim,dim> _mass;
    int _prevIV = -1, _prevIE = -1;
};

/// Assumptions on Spline
/**
  dx =   | V0 | V1 |  E |
      V0 | a  |  0 |  0 |
      V1 | 0  |  a |  0 |
      E  | ***********  |
  */
template<> struct SplineDofs<3,1> {
  static constexpr int dofV = 2, dofE = 0;
};
template<> class Spline<3,1> : public SplineBase<3,1> {
  public:
    const Eigen::Matrix<double,3,4> dx{{0. ,1.,0.,0.},
                                       {0. ,0.,0.,1.},
                                       {-1.,0.,1.,0.}};
    const InterpolatorType& IV(int i, int r) {
      if (_prevIV < 0) {
        _IV[0].resize(1);
        _IV[0][0].resize(1);
        _IV[0][0][0] = {1.,0.}; // Value of the function at the vertex
        _IV[1].resize(2);
        _IV[1][0].resize(0);
        _IV[1][1].resize(1);
        _IV[1][1][0] = {1.,0.}; // Value of the derivative of the function at the vertex
      }
      _prevIV = r;
      return _IV.at(i);
    }
    const InterpolatorType& IE(int i, int r) {return _IE.at(i);}
    double phi(int i, double x) const {
      assert(i >= 0 && i < dim && "Index out of bound");
      const double xs = x/_h;
      switch(i) {
        case 0:
          return 2.*std::pow(xs,3) - 3.*xs*xs + 1.;
        case 1:
          return xs*(xs*xs - 2.*xs + 1.)*_h;
        case 2:
          return xs*xs*(3.-2.*xs);
        case 3:
        default:
          return xs*xs*(xs-1.)*_h;
      }
    }
    double dphi(int i, double x) const {
      assert(i >= 0 && i < dim && "Index out of bound");
      const double xs = x/_h;
      switch(i) {
        case 0:
          return 6.*(xs*xs-xs)/_h;
        case 1:
          return 3.*xs*xs - 4.*xs + 1.;
        case 2:
          return 6.*(xs - xs*xs)/_h;
        case 3:
        default:
          return 3.*xs*xs - 2.*xs;
      }
    }
};
template<> struct SplineDofs<2,0> {
  static constexpr int dofV = 1, dofE = 1;
};
template<> class Spline<2,0> : public SplineBase<2,0> {
  public:
    const Eigen::Matrix<double,2,3> dx{{-1.,1.,0.},
                                       {0. ,1.,-1./_h}};
    const InterpolatorType& IV(int i, int r) {
      if (_prevIV < 0) {
        _IV[0].resize(1);
        _IV[0][0].resize(1);
        _IV[0][0][0] = {1.,0.}; // Value of the function at the vertex
      }
      _prevIV = r;
      return _IV.at(i);
    }
    const InterpolatorType& IE(int i, int r) {
      if (_prevIE != r) {
        _IE[0].resize(1); // Number of derivative to consider
        LegendreGauss quad(r);
        _IE[0][0].resize(quad.npts());
        for (size_t i = 0; i < quad.npts(); ++i) {
          _IE[0][0][i] = {_h*quad.wq(i),_h*quad.tq(i)};
        }
      }
      _prevIE = r;
      return _IE.at(i);
    }
    double phi(int i, double x) const {
      assert(i >= 0 && i < dim && "Index out of bound");
      const double xs = x/_h;
      switch(i) {
        case 0:
          return 3.*xs*xs - 4.*xs + 1;
        case 1:
          return xs*(3.*xs - 2.);
        case 2:
        default:
          return 6.*xs*(1.-xs)/_h;
      }
    }
    double dphi(int i, double x) const {
      assert(i >= 0 && i < dim && "Index out of bound");
      const double xs = x/_h;
      switch(i) {
        case 0:
          return (6.*xs-4.)/_h;
        case 1:
          return (6.*xs - 2.)/_h;
        case 2:
        default:
          return (6. - 12.*xs)/(_h*_h);
      }
    }
};

template<> struct SplineDofs<1,-1> {
  static constexpr int dofV = 0, dofE = 2;
};
template<> class Spline<1,-1> : public SplineBase<1,-1> {
  public:
    const InterpolatorType& IV(int i, int r) {return _IV.at(i);}
    const InterpolatorType& IE(int i, int r) {
      if (_prevIE != r) {
        _IE[0].resize(1); // Number of derivative to consider
        _IE[1].resize(1); // Number of derivative to consider
        LegendreGauss quad(r);
        _IE[0][0].resize(quad.npts());
        _IE[1][0].resize(quad.npts());
        for (size_t i = 0; i < quad.npts(); ++i) {
          _IE[0][0][i] = {_h*quad.wq(i),_h*quad.tq(i)};
          _IE[1][0][i] = {_h*quad.wq(i)*quad.tq(i),_h*quad.tq(i)};
        }
      }
      _prevIE = r;
      return _IE.at(i);
    }
    double phi(int i, double x) const {
      assert(i >= 0 && i < dim && "Index out of bound");
      const double xs = x/_h;
      switch(i) {
        case 0:
          return (4. - 6.*xs)/_h;
        case 1:
        default:
          return (12.*xs - 6.)/_h;
      }
    }
    double dphi(int i, double x) const {
      assert(i >= 0 && i < dim && "Index out of bound");
      switch(i) {
        case 0:
          return -6./(_h*_h);
        case 1:
        default:
          return 12./(_h*_h);
      }
    }
};
#endif
