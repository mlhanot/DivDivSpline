#include "divdiv.hpp"

#include "vectorspace.hpp"

#include <functional>

constexpr std::array<int,6> expandReg(int r1, int r2, int r3) {
  return {r1+2,r1,r2+2,r2,r3+2,r3};
}
constexpr std::array<size_t,6> VDims{3,8,6,3,1,0};

template<size_t D> struct impl_DivDivSpaceBase {};

template<> struct impl_DivDivSpaceBase<0> : public VectorSpace<3,
  expandReg(1,0,0),expandReg(0,1,0),expandReg(0,0,1)> {
    static constexpr size_t degree = 0;
    impl_DivDivSpaceBase(size_t n) : VectorSpace(n) {;}
    using VType = std::tuple_element<degree,ValueTypes>::type;
    using MDiff = Eigen::Matrix<double,VDims[degree+1],Dim>;
    const std::array<VType,Dim> VSpaces{{
      VType{1,0,0},
      VType{0,1,0},
      VType{0,0,1}
    }};
    const Eigen::Matrix<double,Dim,Dim> mass{{ {1,0,0}, {0,1,0}, {0,0,1} }};
    const std::array<MDiff,3> diff{{
      MDiff{{ { 0, 0, 0}, // dx
              { 0, 0, 0},
              { 0, 1, 0},
              { 0, 0, 0},
              { 0, 0, 1},
              { 0, 0, 0},
              { 2./3., 0, 0},
              { 1./3., 0, 0}
      }},
      MDiff{{ { 1, 0, 0}, // dy
              { 0, 0, 0},
              { 0, 0, 0},
              { 0, 0, 0},
              { 0, 0, 0},
              { 0, 0, 1},
              { 0, -1./3., 0},
              { 0, 1./3., 0}
      }},
      MDiff{{ { 0, 0, 0}, // dz
              { 1, 0, 0},
              { 0, 0, 0},
              { 0, 1, 0},
              { 0, 0, 0},
              { 0, 0, 0},
              { 0, 0, -1./3.},
              { 0, 0, -2./3.}
      }}
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

template<> struct impl_DivDivSpaceBase<1> : public VectorSpace<8,
  expandReg(1,-1,0),expandReg(1,0,-1),expandReg(-1,1,0),expandReg(0,1,-1),
  expandReg(-1,0,1),expandReg(0,-1,1),expandReg(0,0,0),expandReg(0,0,0)> {
    static constexpr size_t degree = 1;
    impl_DivDivSpaceBase(size_t n) : VectorSpace(n) {;}
    using VType = std::tuple_element<degree,ValueTypes>::type;
    using MDiff = Eigen::Matrix<double,VDims[degree+1],Dim>;
    const std::array<VType,Dim> VSpaces{{
      VType{{{0,1,0},{0,0,0},{0,0,0}}},
      VType{{{0,0,1},{0,0,0},{0,0,0}}},
      VType{{{0,0,0},{1,0,0},{0,0,0}}},
      VType{{{0,0,0},{0,0,1},{0,0,0}}},
      VType{{{0,0,0},{0,0,0},{1,0,0}}},
      VType{{{0,0,0},{0,0,0},{0,1,0}}},
      VType{{{1,0,0},{0,-1,0},{0,0,0}}},
      VType{{{0,0,0},{0,1,0},{0,0,-1}}}
    }};
    const Eigen::Matrix<double,Dim,Dim> mass{{{1,0,0,0,0,0,0,0},{0,1,0,0,0,0,0,0},{0,0,1,0,0,0,0,0},{0,0,0,1,0,0,0,0},{0,0,0,0,1,0,0,0},{0,0,0,0,0,1,0,0},{0,0,0,0,0,0,2,-1},{0,0,0,0,0,0,-1,2}}};
    const std::array<MDiff,3> diff{{
      MDiff{{{0,0,0,0,0,0,0,0},{0,0,0,-1,0,0,0,0},{0,0,0,0,0,1,0,0},{0,-1.0/2.0,0,0,0,0,0,0},{1.0/2.0,0,0,0,0,0,0,0},{0,0,0,0,0,0,-1.0/2.0,1}}},
      MDiff{{{0,1,0,0,0,0,0,0},{0,0,0,0,0,0,0,0},{0,0,0,0,-1,0,0,0},{0,0,0,1.0/2.0,0,0,0,0},{0,0,0,0,0,0,-1.0/2.0,-1.0/2.0},{0,0,-1.0/2.0,0,0,0,0,0}}},
      MDiff{{{-1,0,0,0,0,0,0,0},{0,0,1,0,0,0,0,0},{0,0,0,0,0,0,0,0},{0,0,0,0,0,0,1,-1.0/2.0},{0,0,0,0,0,-1.0/2.0,0,0},{0,0,0,0,1.0/2.0,0,0,0}}}
    }};
    constexpr static auto interpolateWrapper = std::make_tuple(
      [](auto f, const Eigen::Vector3d &x, unsigned dx, unsigned dy, unsigned dz)->double{
        return f(0,1,x,dx,dy,dz);
      },
      [](auto f, const Eigen::Vector3d &x, unsigned dx, unsigned dy, unsigned dz)->double{
        return f(0,2,x,dx,dy,dz);
      },
      [](auto f, const Eigen::Vector3d &x, unsigned dx, unsigned dy, unsigned dz)->double{
        return f(1,0,x,dx,dy,dz);
      },
      [](auto f, const Eigen::Vector3d &x, unsigned dx, unsigned dy, unsigned dz)->double{
        return f(1,2,x,dx,dy,dz);
      },
      [](auto f, const Eigen::Vector3d &x, unsigned dx, unsigned dy, unsigned dz)->double{
        return f(2,0,x,dx,dy,dz);
      },
      [](auto f, const Eigen::Vector3d &x, unsigned dx, unsigned dy, unsigned dz)->double{
        return f(2,1,x,dx,dy,dz);
      },
      [](auto f, const Eigen::Vector3d &x, unsigned dx, unsigned dy, unsigned dz)->double{
        return f(0,0,x,dx,dy,dz);
      },
      [](auto f, const Eigen::Vector3d &x, unsigned dx, unsigned dy, unsigned dz)->double{
        return -f(2,2,x,dx,dy,dz);
      }
      );
  };

template<> struct impl_DivDivSpaceBase<2> : public VectorSpace<6,
  expandReg(1,-1,-1),expandReg(-1,1,-1),expandReg(-1,-1,1),
  expandReg(0,0,-1),expandReg(0,-1,0),expandReg(-1,0,0)> {
    static constexpr size_t degree = 2;
    impl_DivDivSpaceBase(size_t n) : VectorSpace(n) {;}
    using VType = std::tuple_element<degree,ValueTypes>::type;
    using MDiff = Eigen::Matrix<double,VDims[degree+1],Dim>;
    const std::array<VType,Dim> VSpaces{{
      VType{{{1,0,0},{0,0,0},{0,0,0}}},
      VType{{{0,0,0},{0,1,0},{0,0,0}}},
      VType{{{0,0,0},{0,0,0},{0,0,1}}},
      VType{{{0,1,0},{1,0,0},{0,0,0}}},
      VType{{{0,0,1},{0,0,0},{1,0,0}}},
      VType{{{0,0,0},{0,0,1},{0,1,0}}}
    }};
    const Eigen::Matrix<double,Dim,Dim> mass{{{1,0,0,0,0,0},{0,1,0,0,0,0},{0,0,1,0,0,0},{0,0,0,2,0,0},{0,0,0,0,2,0},{0,0,0,0,0,2}}};
    const std::array<MDiff,3> diff{{
      MDiff{{{1,0,0,0,0,0},{0,0,0,1,0,0},{0,0,0,0,1,0}}},
      MDiff{{{0,0,0,1,0,0},{0,1,0,0,0,0},{0,0,0,0,0,1}}},
      MDiff{{{0,0,0,0,1,0},{0,0,0,0,0,1},{0,0,1,0,0,0}}}
    }};
    constexpr static auto interpolateWrapper = std::make_tuple(
      [](auto f, const Eigen::Vector3d &x, unsigned dx, unsigned dy, unsigned dz)->double{
        return f(0,0,x,dx,dy,dz);
      },
      [](auto f, const Eigen::Vector3d &x, unsigned dx, unsigned dy, unsigned dz)->double{
        return f(1,1,x,dx,dy,dz);
      },
      [](auto f, const Eigen::Vector3d &x, unsigned dx, unsigned dy, unsigned dz)->double{
        return f(2,2,x,dx,dy,dz);
      },
      [](auto f, const Eigen::Vector3d &x, unsigned dx, unsigned dy, unsigned dz)->double{
        return f(0,1,x,dx,dy,dz);
      },
      [](auto f, const Eigen::Vector3d &x, unsigned dx, unsigned dy, unsigned dz)->double{
        return f(0,2,x,dx,dy,dz);
      },
      [](auto f, const Eigen::Vector3d &x, unsigned dx, unsigned dy, unsigned dz)->double{
        return f(1,2,x,dx,dy,dz);
      }
      );
  };

template<> struct impl_DivDivSpaceBase<3> : public VectorSpace<3,
  expandReg(0,-1,-1),expandReg(-1,0,-1),expandReg(-1,-1,0)> {
    static constexpr size_t degree = 3;
    impl_DivDivSpaceBase(size_t n) : VectorSpace(n) {;}
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
      }
      );
  };

template<> struct impl_DivDivSpaceBase<4> : public VectorSpace<1,
  expandReg(-1,-1,-1)> {
    static constexpr size_t degree = 4;
    impl_DivDivSpaceBase(size_t n) : VectorSpace(n) {;}
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

template<size_t D>
DivDivSpace<D>::DivDivSpace(size_t n) : _vspace(std::make_unique<typename decltype(_vspace)::element_type>(n)), 
  mesh(_vspace->mesh()), nbDofs(_vspace->sOffset().back()) {;}

template<size_t D>
DivDivSpace<D>::~DivDivSpace() = default;

template<size_t D>
Eigen::SparseMatrix<double> DivDivSpace<D>::L2() const {
  return _vspace->L2(_vspace->mass);
}

template<size_t D>
Eigen::SparseMatrix<double> DivDivSpace<D>::d() const {
  return _vspace->derivative(_vspace->diff);
}

template<size_t D>
Eigen::VectorXd DivDivSpace<D>::interpolate(std::tuple_element<D,FunctionTypes>::type f, int r) const {
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

template<size_t D>
std::tuple_element<D,ValueTypes>::type DivDivSpace<D>::evaluate(const Eigen::Vector3d &x, const Eigen::Ref<const Eigen::VectorXd> &uh) const {
  return [this,x,uh]<size_t...I>(std::index_sequence<I...>)->decltype(_vspace)::element_type::VType {
    return ((_vspace->template evaluate<I>(x,uh)*_vspace->VSpaces[I])+...);
  }(std::make_index_sequence<decltype(_vspace)::element_type::Dim>());
}

template<size_t D>
Eigen::SparseMatrix<double> DivDivSpace<D>::interiorExtension() const {
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

template<size_t D>
void DivDivSpace<D>::markBoundaryDofs(const Eigen::VectorXi &x) {
  _boundaryDofs = _vspace->markDof(x);
}

template class DivDivSpace<0>;
template class DivDivSpace<1>;
template class DivDivSpace<2>;
template class DivDivSpace<3>;
template class DivDivSpace<4>;
