#ifndef ADM_HPP
#define ADM_HPP

#include "../src/divdiv.hpp"

#include <forward_list>

template<bool useCN>
class ADM {
  public:
    ADM(size_t n, double dt) : _dd0(n), _dd1(n), _dd2(n), _dd3(n), _dd4(n), _offsets([this](){
        std::array<size_t,5> rv;
        rv[0] = 0;
        rv[1] = rv[0] + _dd0.nbDofs;
        rv[2] = rv[1] + _dd1.nbDofs;
        rv[3] = rv[2] + _dd2.nbDofs;
        rv[4] = rv[3] + _dd4.nbDofs;
        return rv;
        }()), _offsetsInterior(_offsets), _offsetsBoundary({0,0,0,0,0}), _boundaryMarked(false) {
      setupMass();
      setupA(dt);
    }
    /// Change the timestep dt
    void updateTimestep(double dt) {setupA(dt);}
    /// Get the offset (including boundary) of the spaces
    size_t offset(size_t i) const {return _offsets[i];}
    /// Return the bilinear form associated with the system
    Eigen::SparseMatrix<double> system() const {
      if (_boundaryMarked) {
	if constexpr(useCN) return _interiorExt.transpose()*(_M - _L)*_interiorExt;
        else return _interiorExt.transpose()*_L*_interiorExt;
      } else {
	if constexpr(useCN) return _M-_L;
        else return _L;
      }
    }
    /// Assemble the RHS
    Eigen::VectorXd RHS(const Eigen::VectorXd &uOld, const Eigen::VectorXd &ub, const Eigen::VectorXd &ubOld) const {
      if constexpr(useCN) {
	if (_boundaryMarked) {
          return _interiorExt.transpose()*(-(_M-_L)*_boundaryExt*ub + (_M+_L)*(_interiorExt*uOld + _boundaryExt*ubOld));
        } else {
          return _interiorExt.transpose()*(_M+_L)*_interiorExt*uOld;
        }
      } else {
        if (_boundaryMarked) {
	  return _interiorExt.transpose()*(-_L*_boundaryExt*ub + _M*(_interiorExt*uOld + _boundaryExt*ubOld));
        } else {
          return _interiorExt.transpose()*_M*_interiorExt*uOld;
        }
      }
    };
    /// Interpolate the interior degrees of freedom
    template<typename Sol>
    void interpolate(Eigen::VectorXd &u, Sol sol, int r = -1) const {
      assert(static_cast<size_t>(u.size()) == _offsets[4]);
      u.segment(_offsets[1],_offsets[2]-_offsets[1]) = _dd1.interpolate(std::bind_front(&Sol::f1,&sol),r);
      u.segment(_offsets[2],_offsets[3]-_offsets[2]) = _dd2.interpolate(std::bind_front(&Sol::f2,&sol),r);
    }
    /// Returns the mass matrices of the indivdual spaces
    Eigen::SparseMatrix<double> mass(size_t i) const {
      assert(i < 4);
      switch(i) {
        case 0: return _dd0.L2();
        case 1: return _dd1.L2();
        case 2: return _dd2.L2();
        default: return _dd4.L2();
      }
    }
    /// Return the mass matrix of the global system
    const Eigen::SparseMatrix<double>& mass() const {return _M;}
    const Eigen::SparseMatrix<double>& interiorExt() const {return _interiorExt;}
    const Eigen::SparseMatrix<double>& boundaryExt() const {return _boundaryExt;}
    /// Mark all boundary
    void markAllBoundary() {
      markBoundary(_dd0.allBoundary);
    }
    /// Mark boundary elements
    template<typename F> void markBoundary(F f) {
      _dd0.markBoundary(f);
      _dd1.markBoundary(f);
      _dd2.markBoundary(f);
      _dd4.markBoundary(f);
      { // Setup interiorExt and _offsetsInterior/Boundary
        std::forward_list<Eigen::Triplet<double>> triplets;
        auto setup = [this,&triplets](Eigen::SparseMatrix<double> B,size_t i) {
          assert(i > 0 && i < 5);
          _offsetsInterior[i] = _offsetsInterior[i-1] + B.cols();
          _offsetsBoundary[i] = _offsetsBoundary[i-1] + B.rows() - B.cols();
          blockAssemble(triplets,_offsets[i-1],_offsetsInterior[i-1],B);
        };
        setup(_dd0.interiorExtension(),1);
        setup(_dd1.interiorExtension(),2);
        setup(_dd2.interiorExtension(),3);
        setup(_dd4.interiorExtension(),4);
        Eigen::SparseMatrix<double> rv(_offsets[4],_offsetsInterior[4]);
        rv.setFromTriplets(triplets.begin(),triplets.end());
        _interiorExt = rv;
      }
      { // Setup boundaryExt
        std::forward_list<Eigen::Triplet<double>> triplets;
        auto setup = [this,&triplets](Eigen::SparseMatrix<double> B,size_t i) {
          assert(i > 0 && i < 5);
          blockAssemble(triplets,_offsets[i-1],_offsetsBoundary[i-1],B);
        };
        setup(_dd0.boundaryExtension(),1);
        setup(_dd1.boundaryExtension(),2);
        setup(_dd2.boundaryExtension(),3);
        setup(_dd4.boundaryExtension(),4);
        Eigen::SparseMatrix<double> rv(_offsets[4],_offsetsBoundary[4]);
        rv.setFromTriplets(triplets.begin(),triplets.end());
        _boundaryExt = rv;
      }
      _boundaryMarked = true;
    }
  private:
    static void blockAssemble(std::forward_list<Eigen::Triplet<double>> &triplets, size_t offsetR, size_t offsetC, const Eigen::SparseMatrix<double> &A) {
      for (int k=0; k<A.outerSize(); ++k) {
        for (Eigen::SparseMatrix<double>::InnerIterator it(A,k); it; ++it) {
          triplets.emplace_front(offsetR+it.row(),offsetC+it.col(),it.value());
        }
      }
    }
    void setupMass() {
      Eigen::SparseMatrix<double> rv(_offsets[4],_offsets[4]);
      std::forward_list<Eigen::Triplet<double>> triplets;
      blockAssemble(triplets,_offsets[0],_offsets[0],_dd0.L2());
      blockAssemble(triplets,_offsets[1],_offsets[1],_dd1.L2());
      blockAssemble(triplets,_offsets[2],_offsets[2],_dd2.L2());
      blockAssemble(triplets,_offsets[3],_offsets[3],_dd4.L2());
      rv.setFromTriplets(triplets.begin(),triplets.end());
      _M = rv;
    }
    void setupA(double dt) {
      std::forward_list<Eigen::Triplet<double>> triplets;
      blockAssemble(triplets,_offsets[1],_offsets[0],-_dd0.d());
      blockAssemble(triplets,_offsets[2],_offsets[1],_dd1.d());
      blockAssemble(triplets,_offsets[3],_offsets[2],-_dd3.d()*_dd2.d());
      Eigen::SparseMatrix<double> Jd(_offsets[4],_offsets[4]);
      Jd.setFromTriplets(triplets.begin(),triplets.end());
      if constexpr(useCN) _L =  0.5*dt*(_M*Jd - Jd.transpose()*_M);
      else  _L = _M - dt*_M*Jd + dt*Jd.transpose()*_M;
    }
  private:
    DivDivSpace<0> _dd0;
    DivDivSpace<1> _dd1;
    DivDivSpace<2> _dd2;
    DivDivSpace<3> _dd3;
    DivDivSpace<4> _dd4;
    const std::array<size_t,5> _offsets;
    std::array<size_t,5> _offsetsInterior, _offsetsBoundary;
    Eigen::SparseMatrix<double> _M, _L, _interiorExt, _boundaryExt;
    bool _boundaryMarked;
};

#endif
