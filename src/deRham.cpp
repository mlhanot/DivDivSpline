#include "deRham.hpp"

#include "vectorspace.hpp"

#include <functional>

constexpr std::array<int,6> expandReg(int r1, int r2, int r3) {
  return {r1+2,r1,r2+2,r2,r3+2,r3};
}
constexpr std::array<size_t,6> VDims{1,3,3,1,0};
template<size_t D,int _r>
using implParent = std::tuple_element<D,std::tuple<
VectorSpace<1,expandReg(0+_r,0+_r,0+_r)>,
VectorSpace<3,expandReg(-1+_r,0+_r,0+_r), expandReg(0+_r,-1+_r,0+_r), expandReg(0+_r,0+_r,-1+_r)>,
VectorSpace<3,expandReg(0+_r,-1+_r,-1+_r), expandReg(-1+_r,0+_r,-1+_r), expandReg(-1+_r,-1+_r,0+_r)>,
VectorSpace<1,expandReg(-1+_r,-1+_r,-1+_r)>
>>::type;

template<size_t D,int _r> struct impl_DeRhamSpaceBase {};

template<int _r> struct impl_DeRhamSpaceBase<0,_r>: public implParent<0,_r>  {
    static constexpr size_t degree = 0;
    static constexpr size_t Dim = VDims[degree];
    impl_DeRhamSpaceBase(size_t n) : implParent<degree,_r>(n) {;}
    using VType = std::tuple_element<degree,ValueTypes>::type;
    using MDiff = Eigen::Matrix<double,VDims[degree+1],Dim>;
    const std::array<VType,Dim> VSpaces{{
      VType{1}
    }};
    const Eigen::Matrix<double,Dim,Dim> mass{1};
    const std::array<MDiff,3> diff{{
      MDiff{1},
      MDiff{0},
      MDiff{0}
    }};
    constexpr static auto interpolateWrapper = std::make_tuple(
      [](auto f, const Eigen::Vector3d &x, unsigned dx, unsigned dy, unsigned dz)->double{
        return f(x,dx,dy,dz);
      });
  };

template<int _r> struct impl_DeRhamSpaceBase<1,_r> : public implParent<1,_r> {
    static constexpr size_t degree = 1;
    static constexpr size_t Dim = VDims[degree];
    impl_DeRhamSpaceBase(size_t n) : implParent<degree,_r>(n) {;}
    using VType = std::tuple_element<degree,ValueTypes>::type;
    using MDiff = Eigen::Matrix<double,VDims[degree+1],Dim>;
    const std::array<VType,Dim> VSpaces{{
      VType{1,0,0},
      VType{0,1,0},
      VType{0,0,1}
    }};
    const Eigen::Matrix<double,Dim,Dim> mass{{{1,0,0},{0,1,0},{0,0,1}}};
    const std::array<MDiff,3> diff{{
      MDiff{{{0,0,0},{0,0,-1},{0,1,0}}},
      MDiff{{{0,0,1},{0,0,0},{-1,0,0}}},
      MDiff{{{0,-1,0},{1,0,0},{0,0,0}}}
    }};
    constexpr static auto interpolateWrapper = std::make_tuple(
      [](auto f, const Eigen::Vector3d &x, unsigned dx, unsigned dy, unsigned dz)->double{
        return f(0,x,dx,dy,dz);
      },
      [](auto f, const Eigen::Vector3d &x, unsigned dx, unsigned dy, unsigned dz)->double{
        return f(1,x,dx,dy,dz);
      },
      [](auto f, const Eigen::Vector3d &x, unsigned dx, unsigned dy, unsigned dz)->double{
        return f(2,x,dx,dy,dz);
      });
  };

template<int _r> struct impl_DeRhamSpaceBase<2,_r> : public implParent<2,_r> {
    static constexpr size_t degree = 2;
    static constexpr size_t Dim = VDims[degree];
    impl_DeRhamSpaceBase(size_t n) : implParent<degree,_r>(n) {;}
    using VType = std::tuple_element<degree,ValueTypes>::type;
    using MDiff = Eigen::Matrix<double,VDims[degree+1],Dim>;
    const std::array<VType,Dim> VSpaces{{
      VType{1,0,0},
      VType{0,1,0},
      VType{0,0,1}
    }};
    const Eigen::Matrix<double,Dim,Dim> mass{{{1,0,0},{0,1,0},{0,0,1}}};
    const std::array<MDiff,3> diff{{
      MDiff{1,0,0},
      MDiff{0,1,0},
      MDiff{0,0,1}
    }};
    constexpr static auto interpolateWrapper = std::make_tuple(
      [](auto f, const Eigen::Vector3d &x, unsigned dx, unsigned dy, unsigned dz)->double{
        return f(0,x,dx,dy,dz);
      },
      [](auto f, const Eigen::Vector3d &x, unsigned dx, unsigned dy, unsigned dz)->double{
        return f(1,x,dx,dy,dz);
      },
      [](auto f, const Eigen::Vector3d &x, unsigned dx, unsigned dy, unsigned dz)->double{
        return f(2,x,dx,dy,dz);
      });
  };

template<int _r> struct impl_DeRhamSpaceBase<3,_r> : public implParent<3,_r> {
    static constexpr size_t degree = 3;
    static constexpr size_t Dim = VDims[degree];
    impl_DeRhamSpaceBase(size_t n) : implParent<degree,_r>(n) {;}
    using VType = std::tuple_element<degree,ValueTypes>::type;
    using MDiff = Eigen::Matrix<double,VDims[degree+1],Dim>;
    const std::array<VType,Dim> VSpaces {{
      VType{1}
    }};
    const Eigen::Matrix<double,Dim,Dim> mass{{{1}}};
    const std::array<MDiff,3> diff;
    constexpr static auto interpolateWrapper = std::make_tuple(
      [](auto f, const Eigen::Vector3d &x, unsigned dx, unsigned dy, unsigned dz)->double{
        return f(x,dx,dy,dz);
      }
      );
  };

// ----------------------------------------------------------------------------------------------------------

template<size_t D,int _r>
DeRhamSpace<D,_r>::DeRhamSpace(size_t n) : _vspace(std::make_unique<typename decltype(_vspace)::element_type>(n)), 
  mesh(_vspace->mesh()), nbDofs(_vspace->sOffset().back()), SDim(_vspace->Dim) {;}

template<size_t D,int _r>
DeRhamSpace<D,_r>::~DeRhamSpace() = default;

template<size_t D,int _r>
Eigen::SparseMatrix<double> DeRhamSpace<D,_r>::L2() const {
  return _vspace->L2(_vspace->mass);
}

template<size_t D,int _r>
Eigen::SparseMatrix<double> DeRhamSpace<D,_r>::d() const {
  return _vspace->derivative(_vspace->diff);
}

template<size_t D,int _r>
Eigen::VectorXd DeRhamSpace<D,_r>::interpolate(std::tuple_element<D,FunctionTypes>::type f, int r) const {
  Eigen::VectorXd rv(nbDofs);
  constexpr size_t Dim = decltype(_vspace)::element_type::Dim;
  const auto fTuple = [this,f]<size_t...I>(std::index_sequence<I...>) {
    return std::make_tuple(std::bind_front(std::get<I>(_vspace->interpolateWrapper),f)...);
  }(std::make_index_sequence<Dim>());
  [this,&r,&rv,&fTuple]<size_t...I>(std::index_sequence<I...>) {
    (_vspace->template interpolate<I,decltype(std::get<I>(fTuple))>(rv,std::get<I>(fTuple),r),...);
  }(std::make_index_sequence<Dim>());
  return rv;
}

template<size_t D,int _r>
std::tuple_element<D,ValueTypes>::type DeRhamSpace<D,_r>::evaluate(const Eigen::Vector3d &x, const Eigen::Ref<const Eigen::VectorXd> &uh) const {
  return [this,x,uh]<size_t...I>(std::index_sequence<I...>)->decltype(_vspace)::element_type::VType {
    return ((_vspace->template evaluate<I>(x,uh)*_vspace->VSpaces[I])+...);
  }(std::make_index_sequence<decltype(_vspace)::element_type::Dim>());
}

template<size_t D,int _r>
Eigen::SparseMatrix<double> DeRhamSpace<D,_r>::interiorExtension() const {
  if (static_cast<size_t>(_boundaryDofs.size()) != nbDofs) {
    assert(_boundaryDofs.size() == 0); // No boundary was marked, returns identity
    Eigen::SparseMatrix<double> rv(nbDofs,nbDofs);
    std::forward_list<Eigen::Triplet<double>> triplets;
    for (size_t iDof = 0; iDof < nbDofs; ++iDof) {
      triplets.emplace_front(iDof,iDof,1.);
    }
    rv.setFromTriplets(triplets.begin(),triplets.end());
    return rv;
  }
  size_t accI = 0;
  std::forward_list<Eigen::Triplet<double>> triplets;
  for (size_t iDof = 0; iDof < nbDofs; ++iDof) {
    if (_boundaryDofs[iDof] > 0) continue;
    triplets.emplace_front(iDof,accI++,1.);
  }
  Eigen::SparseMatrix<double> rv(nbDofs,accI);
  rv.setFromTriplets(triplets.begin(),triplets.end());
  return rv;
}

template<size_t D,int _r>
void DeRhamSpace<D,_r>::markBoundaryDofs(const Eigen::VectorXi &x) {
  _boundaryDofs = _vspace->markDof(x);
}
template<size_t D,int _r>
size_t DeRhamSpace<D,_r>::sOffset(size_t i) const {
  return _vspace->sOffset().at(i);
}

template class DeRhamSpace<0,0>;
template class DeRhamSpace<1,0>;
template class DeRhamSpace<2,0>;
template class DeRhamSpace<3,0>;
template class DeRhamSpace<0,1>;
template class DeRhamSpace<1,1>;
template class DeRhamSpace<2,1>;
template class DeRhamSpace<3,1>;
