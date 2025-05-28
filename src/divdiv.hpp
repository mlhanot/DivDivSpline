#ifndef DIVDIV_HPP
#define DIVDIV_HPP

#include "mesh.hpp"

#include <tuple>
#include <Eigen/Sparse>
#include <memory>

using FunctionTypes = std::tuple<
  std::function<double(unsigned,const Eigen::Vector3d &,unsigned,unsigned,unsigned)>, // H1 (R^3)
  std::function<double(unsigned,unsigned,const Eigen::Vector3d &,unsigned,unsigned,unsigned)>, // H(symcurl) (R^{3x3})
  std::function<double(unsigned,unsigned,const Eigen::Vector3d &,unsigned,unsigned,unsigned)>, // H(divdiv) (R^{3x3})
  std::function<double(unsigned,const Eigen::Vector3d &,unsigned,unsigned,unsigned)>, // H(div) (R^3)
  std::function<double(const Eigen::Vector3d &,unsigned,unsigned,unsigned)> // L2 (R)
  >; 
using ValueTypes = std::tuple<
  Eigen::Vector3d,
  Eigen::Matrix3d,
  Eigen::Matrix3d,
  Eigen::Vector3d,
  double
  >;

/// Merge the components into a vector/matrix
template<size_t degree,typename F> requires(degree < 5 && std::convertible_to<F,typename std::tuple_element<degree,FunctionTypes>::type>)
std::tuple_element<degree,ValueTypes>::type mergeComp(F f, const Eigen::Vector3d &x, unsigned dx, unsigned dy, unsigned dz) {
  typename std::tuple_element<degree,ValueTypes>::type rv;
  if constexpr (degree == 0) {
    for (size_t i = 0; i < 3; ++i) {
      rv(i) = f(i,x,dx,dy,dz);
    }
  } else if constexpr (degree < 3) {
    for (size_t i = 0; i < 3; ++i) {
      for (size_t j = 0; j < 3; ++j) {
        rv(i,j) = f(i,j,x,dx,dy,dz);
      }
    }
  } else {
    rv = f(x,dx,dy,dz);
  }
  return rv;
};

template<size_t> struct impl_DivDivSpaceBase;
/// Spaces for the divdiv complex
/**
  \warning The differential operator is always first order. The divdiv operator is obtained composing the 2nd and 3rd diff
  */
template<size_t D> requires(D < 5)
class DivDivSpace {
  private:
    const std::unique_ptr<impl_DivDivSpaceBase<D>> _vspace;
  public:
    DivDivSpace(size_t n);
    ~DivDivSpace();
    const Mesh mesh;
    const size_t nbDofs;
    Eigen::SparseMatrix<double> L2() const;
    Eigen::SparseMatrix<double> d() const;
    Eigen::VectorXd interpolate(std::tuple_element<D,FunctionTypes>::type f, int r = -1) const;
    std::tuple_element<D,ValueTypes>::type evaluate(const Eigen::Vector3d &x, const Eigen::Ref<const Eigen::VectorXd> &uh) const;
    /// Marks boundary elements
    template<typename F> void markBoundary(F f) {
      markBoundaryDofs(mesh.markBoundary(f));
    }
    /// Helper to mark all the boundary 
    static bool allBoundary(const Eigen::Vector3d &x) {
      constexpr double esp = 1e-10;
      return (x[0] < esp) || (x[1] < esp) || (x[2] < esp) || (x[0]-1. < esp) || (x[1]-1. < esp) || (x[2]-1. < esp);
    };
    /// Returns the extension operator K(nbDofs,nbDofs-boundaryDim)
    Eigen::SparseMatrix<double> interiorExtension() const;
  private:
    Eigen::VectorXi _boundaryDofs;
    void markBoundaryDofs(const Eigen::VectorXi &);
};

extern template class DivDivSpace<0>;
extern template class DivDivSpace<1>;
extern template class DivDivSpace<2>;
extern template class DivDivSpace<3>;
extern template class DivDivSpace<4>;

#endif
