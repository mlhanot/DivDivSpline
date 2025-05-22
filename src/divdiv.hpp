#ifndef DIVDIV_HPP
#define DIVDIV_HPP

#include "mesh.hpp"

#include <tuple>
#include <Eigen/Sparse>
#include <memory>

template<size_t D> struct impl_DivDivSpaceBase;

using FunctionTypes = std::tuple<
  std::function<double(unsigned,const Eigen::Vector3d &,unsigned,unsigned,unsigned)>, // H1 (R^3)
  std::function<double(unsigned,unsigned,const Eigen::Vector3d &,unsigned,unsigned,unsigned)>, // H(symcurl) (R^{3x3})
  std::function<double(unsigned,unsigned,const Eigen::Vector3d &,unsigned,unsigned,unsigned)>, // H(divdiv) (R^{3x3})
  std::function<double(const Eigen::Vector3d &,unsigned,unsigned,unsigned)>>; // L2 (R)
using ValueTypes = std::tuple<
  Eigen::Vector3d,
  Eigen::Matrix3d,
  Eigen::Matrix3d,
  double>;

template<size_t D> requires(D < 4)
class DivDivSpace {
  private:
    std::unique_ptr<impl_DivDivSpaceBase<D>> _vspace;
  public:
    DivDivSpace(size_t n);
    ~DivDivSpace();
    const Mesh mesh;
    const size_t nbDofs;
    Eigen::SparseMatrix<double> L2() const;
    Eigen::SparseMatrix<double> d() const;
    Eigen::VectorXd interpolate(std::tuple_element<D,FunctionTypes>::type f, int r = -1) const;
    std::tuple_element<D,ValueTypes>::type evaluate(const Eigen::Vector3d &x, const Eigen::Ref<const Eigen::VectorXd> &uh) const;
};

extern template class DivDivSpace<0>;
extern template class DivDivSpace<1>;
extern template class DivDivSpace<2>;
extern template class DivDivSpace<3>;

#endif
