#ifndef DERHAM_HPP
#define DERHAM_HPP

#include "mesh.hpp"

#include <tuple>
#include <Eigen/Sparse>
#include <memory>

using FunctionTypes = std::tuple<
  std::function<double(const Eigen::Vector3d &,unsigned,unsigned,unsigned)>, // H1 
  std::function<double(unsigned,const Eigen::Vector3d &,unsigned,unsigned,unsigned)>, // H(curl) (R^3)
  std::function<double(unsigned,const Eigen::Vector3d &,unsigned,unsigned,unsigned)>, // H(div) (R^3)
  std::function<double(const Eigen::Vector3d &,unsigned,unsigned,unsigned)> // L2 (R)
  >; 
using ValueTypes = std::tuple<
  double,
  Eigen::Vector3d,
  Eigen::Vector3d,
  double
  >;

/// Merge the components into a vector/matrix
template<size_t degree,typename F> requires(degree < 4 && std::convertible_to<F,typename std::tuple_element<degree,FunctionTypes>::type>)
std::tuple_element<degree,ValueTypes>::type mergeComp(F f, const Eigen::Vector3d &x, unsigned dx, unsigned dy, unsigned dz) {
  typename std::tuple_element<degree,ValueTypes>::type rv;
  if constexpr (degree == 1 || degree == 2) {
    for (size_t i = 0; i < 3; ++i) {
      rv(i) = f(i,x,dx,dy,dz);
    }
  } else {
    rv = f(x,dx,dy,dz);
  }
  return rv;
};

template<size_t,int> struct impl_DeRhamSpaceBase;
/// Spaces for the de Rham complex
template<size_t D,int _r> requires(D < 4 && _r >= 0 && _r < 2)
class DeRhamSpace {
  private:
    const std::unique_ptr<impl_DeRhamSpaceBase<D,_r>> _vspace;
  public:
    DeRhamSpace(size_t n);
    ~DeRhamSpace();
    const Mesh mesh;
    const size_t nbDofs;
    const size_t SDim; /// Number of spline components
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
      return (x[0] < esp) || (x[1] < esp) || (x[2] < esp) || (1-x[0] < esp) || (1-x[1] < esp) || (1-x[2] < esp);
    };
    /// Returns the extension operator K(nbDofs,nbDofs-boundaryDim)
    Eigen::SparseMatrix<double> interiorExtension() const;
    /// Return the offset of the individual splines components
    size_t sOffset(size_t i) const;
  private:
    Eigen::VectorXi _boundaryDofs;
    void markBoundaryDofs(const Eigen::VectorXi &);
};


extern template class DeRhamSpace<0,0>;
extern template class DeRhamSpace<1,0>;
extern template class DeRhamSpace<2,0>;
extern template class DeRhamSpace<3,0>;
extern template class DeRhamSpace<0,1>;
extern template class DeRhamSpace<1,1>;
extern template class DeRhamSpace<2,1>;
extern template class DeRhamSpace<3,1>;

#endif
