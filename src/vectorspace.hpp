#ifndef VECTORSPACE_HPP
#define VECTORSPACE_HPP

#include "tensorspline.hpp"

template<size_t Dim>
auto initTuple(size_t n) {
  return std::tuple_cat(initTuple<Dim-1>(n),std::make_tuple(n));
};
template<>
auto initTuple<1>(size_t n) {
  return std::make_tuple(n);
};

template<size_t _Dim, std::array<int,6> ...TSpaces>
requires(sizeof...(TSpaces) == _Dim && _Dim > 0)
class VectorSpace {
  public:
    static constexpr size_t Dim = _Dim;
    VectorSpace(size_t n) : _Splines(initTuple<Dim>(n)), 
    _sOffset([](std::array<size_t,Dim> a){
        std::array<size_t,Dim+1> rv;
        rv[0] = 0;
        for (size_t i = 0; i < Dim; ++i) {
          rv[i+1] = rv[i] + a[i];
        }
        return rv;
      }(std::apply([](auto&... sp)->std::array<size_t,Dim>{
        return {sp.nbDofs...};
      },_Splines)))
      {;}
    const Mesh& mesh() const {return std::get<0>(_Splines).mesh();}
    const std::array<size_t,Dim+1>& sOffset() const {return _sOffset;}
    /// Generic function to compute the sum of the Kronecker product
    template<int RDim>
    Eigen::SparseMatrix<double> Kronecker(const std::array<size_t,RDim+1> &RsOffset,
        const Eigen::Matrix<double,RDim,Dim> &D /*!< Operator on the image space */, 
        std::array<Eigen::SparseMatrix<double>,Dim> &&M /*!< List of individual matrices */) const {
      constexpr double eps = 1e-14;
      Eigen::SparseMatrix<double> rv(RsOffset[RDim],sOffset()[Dim]);
      std::forward_list<Eigen::Triplet<double>> triplets;
      for (size_t r = 0; r < RDim; ++r) {
        for (size_t c = 0; c < Dim; ++c) {
          const double Dval = D(r,c);
          if (std::abs(Dval) < eps) continue;
          for (int k=0; k<M[c].outerSize(); ++k) {
            for (Eigen::SparseMatrix<double>::InnerIterator it(M[c],k); it; ++it) {
              triplets.emplace_front(RsOffset[r]+it.row(),sOffset()[c]+it.col(),Dval*it.value());
            }
          }
        }
      }
      rv.setFromTriplets(triplets.begin(),triplets.end());
      return rv;
    }
    /// Specialization to compute the differential
    template<int RDim>
    Eigen::SparseMatrix<double> derivative(const std::array<Eigen::Matrix<double,RDim,Dim>,3> &D) const {
      // Compute the target space dimension
      std::array<size_t,RDim+1> RsOffset;
      RsOffset.fill(0);
      auto fillIfNz = [this,&RsOffset,&D]<size_t c, size_t dir>(size_t r) {
        if (RsOffset[r+1] > 0) return;
        if (std::abs(D[dir](r,c)) < 1e-14) return;
        TensorSpline<std::tuple_element<c,decltype(_Splines)>::type::template derivativeSpace<dir>()> Sp(mesh().Nx);
        RsOffset[r+1] = Sp.nbDofs;
      };
      auto dirIterate = [&fillIfNz]<size_t c, size_t... I2>(size_t r, std::integral_constant<size_t,c>, std::index_sequence<I2...>) {
        (fillIfNz.template operator()<c,I2>(r),...);
      };
      auto cIterate = [&dirIterate]<size_t... I1>(size_t r, std::index_sequence<I1...>) {
          (dirIterate(r,std::integral_constant<size_t,I1>{},std::make_index_sequence<3>()),...);
      };
      for (size_t r = 0; r < RDim; ++r) {
        cIterate(r, std::make_index_sequence<Dim>());
        assert(RsOffset[r+1] > 0 && "Could not determine the image space dimension");
        RsOffset[r+1] += RsOffset[r];
      }

      auto KrDiv = [this,&RsOffset,&D]<size_t dir>()->Eigen::SparseMatrix<double> {
        return Kronecker(RsOffset,D[dir],std::apply([](auto&... ts)->std::array<Eigen::SparseMatrix<double>,Dim>{
                return {ts.template derivative<dir>()...};
               },_Splines));
      };
      return [&KrDiv]<size_t... I>(std::index_sequence<I...>)->Eigen::SparseMatrix<double>{
        return (KrDiv.template operator()<I>() + ...);
      }(std::make_index_sequence<3>());
    }
    /// Specialization to compute the mass
    Eigen::SparseMatrix<double> L2(const Eigen::Matrix<double,Dim,Dim> &D) const {
      return Kronecker(_sOffset,D,std::apply([](auto&... ts)->std::array<Eigen::SparseMatrix<double>,Dim>{
            return {ts.L2()...};
          },_Splines));
    }
    /// Interpolate on the i-th component
    template<size_t iS, scalarField F>
      requires(iS < Dim)
    void interpolate(Eigen::VectorXd &uh /*!< Output vector, the correct segment for the given component is automaticaly determined */, 
        F f /*!< Function to interpolate, it must return a scalar value*/,
        int r = -1 /*!< Interpolation degree */) {
      assert(static_cast<size_t>(uh.size()) == _sOffset[Dim]);
      size_t acc = _sOffset[iS];
      for (size_t iDim = 0; iDim < 4; ++iDim) {
        for (size_t iT = 0; iT < mesh().nbC[iDim]; ++iT) {
          const size_t localSize = std::get<iS>(_Splines).localDim(iDim,iT);
          for (size_t i = 0; i < localSize; ++i) {
            uh[acc++] = std::get<iS>(_Splines).interpolate(f,iDim,iT,i,r);
          }
        }
      }
    }
    template<size_t i> requires(i < Dim)
    double evaluate(const Eigen::Vector3d &x, const Eigen::VectorXd &uh) const {
      assert(static_cast<size_t>(uh.size()) == _sOffset[Dim]);
      return std::get<i>(_Splines).evaluate(x,uh.segment(_sOffset[i],_sOffset[i+1]-_sOffset[i]));
    }
  private:
    std::tuple<TensorSpline<TSpaces>...> _Splines;
    const std::array<size_t,Dim+1> _sOffset;
};

#endif
