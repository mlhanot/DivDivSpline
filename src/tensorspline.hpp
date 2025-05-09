#ifndef TENSORSPLINE_HPP
#define TENSORSPLINE_HPP

#include "spline.hpp"
#include "mesh.hpp"

/// Class for the tensor product of three splines
template<int _d1, int _r1, int _d2, int _r2, int _d3, int _r3>
class TensorSpline {
  public:
    TensorSpline(size_t n) : _h(1./n), _S1(_h), _S2(_h), _S3(_h), _mesh(n),
      gOffsetEx(_mesh.nbC[0]*dofV),
      gOffsetEy(gOffsetEx + _mesh.IEX*dofE[0]),
      gOffsetEz(gOffsetEy + _mesh.IEX*dofE[1]),
      gOffsetFz(gOffsetEz + _mesh.IEX*dofE[2]),
      gOffsetFx(gOffsetFz + _mesh.IFX*dofF[0]),
      gOffsetFy(gOffsetFx + _mesh.IFX*dofF[1]),
      gOffsetT(gOffsetFy + _mesh.IFX*dofF[2])
    {}
    /// Number of degree of freedoms associated to each element. 
    /**
      \warning The number associated to edges and faces varies depending on the orientation
    */
    static constexpr size_t dofV = _S1.dofV*_S2.dofV*_S3.dofV;
    static constexpr std::array<size_t,3> dofE{_S1.dofE*_S2.dofV*_S3.dofV,
                                               _S1.dofV*_S2.dofE*_S3.dofV,
                                               _S1.dofV*_S2.dofV*_S3.dofE}; // Along x, y, then z
    static constexpr std::array<size_t,3> dofF{_S1.dofE*_S2.dofE*_S3.dofV,
                                               _S1.dofV*_S2.dofE*_S3.dofE,
                                               _S1.dofE*_S2.dofV*_S3.dofE}; // Along z, x, then y
    static constexpr size_t dofT = _S1.dofE*_S2.dofE*_S3.dofE;
    /// Global offsets of the corresponding elements (in the spline space)
    const size_t gOffsetEx, gOffsetEy, gOffsetEz, gOffsetFz, gOffsetFx, gOffsetFy, gOffsetT;
    /// Convert the local index of the tensor product to the locals indices of each components
    static constexpr std::array<int,3> splitIndex(int i,int S1, int S2, int S3) {
      if (S1*S2*S3 == 0) return {-1,-1,-1};
      const int i2 = i%S1, ix = i/S1;
      const int iz = i2%S2, iy = i2/S2;
      return {ix,iy,iz};
    }
    /// Convert the local index on a cell to the local index in the splines components
    std::array<int,3> splitBasis(size_t Fdim /*!< Entity dimension */,
        size_t iFT /*!< Local index of the entity within the cell */, int ind /*!< Local index of the dof associated with iFT */) const {
      assert(Fdim < 4);
      assert(ind >= 0);
      if (Fdim == 0) {
        assert(ind < dofV);
        std::array<size_t,3> splitI = splitIndex(ind,_S1.dofV,_S2.dofV,_S3.dofV);
        splitI[0] += _mesh.iVTtoSide[iFT][0]*_S1.dofV;
        splitI[1] += _mesh.iVTtoSide[iFT][1]*_S2.dofV;
        splitI[2] += _mesh.iVTtoSide[iFT][2]*_S3.dofV;
        return splitI;
      } else if (Fdim == 1) {
        const size_t Axis = _mesh.iETtoAxis[iFT];
        assert(ind < dofE[Axis]);
        std::array<size_t,3> splitI = splitIndex(ind,(Axis==0)?_S1.dofE:_S1.dofV,(Axis==1)?_S2.dofE:_S2.dofV,(Axis==2)?_S3.dofE:_S3.dofV);
        splitI[0] += (_mesh.iETtoSide[iFT][0]+(Axis==0?2:0))*_S1.dofV;
        splitI[1] += (_mesh.iETtoSide[iFT][1]+(Axis==1?2:0))*_S2.dofV;
        splitI[2] += (_mesh.iETtoSide[iFT][2]+(Axis==2?2:0))*_S3.dofV;
        return splitI;
      } else if (Fdim == 2) {
        const size_t Axis = _mesh.iFTtoAxis[iFT];
        assert(ind < dofF[Axis]);
        std::array<size_t,3> splitI = splitIndex(ind,(Axis!=0)?_S1.dofE:_S1.dofV,(Axis!=1)?_S2.dofE:_S2.dofV,(Axis!=2)?_S3.dofE:_S3.dofV);
        splitI[0] += (_mesh.iFTtoSide[iFT][0]+(Axis!=0?2:0))*_S1.dofV;
        splitI[1] += (_mesh.iFTtoSide[iFT][1]+(Axis!=1?2:0))*_S2.dofV;
        splitI[2] += (_mesh.iFTtoSide[iFT][2]+(Axis!=2?2:0))*_S3.dofV;
        return splitI;
      } else {
        assert(Fdim == 3);
        assert(ind < dofT);
        std::array<size_t,3> splitI = splitIndex(ind,_S1.dofE,_S2.dofE,_S3.dofE);
        splitI[0] += 2*_S1.dofV;
        splitI[1] += 2*_S2.dofV;
        splitI[2] += 2*_S3.dofV;
        return splitI;
      }
    }
    /// Type of the function to interpolate. The suitable component must be extracted. The inputs dx,dy and dz
    template<typename T> concept scalarField = requires(T f, const Eigen::Vector3d &x, unsigned dx, unsigned dy, unsigned dz) {{f(x,dx,dy,dz)} -> std::convertible_to<double>;};
    /// Interpolate a scalar field
    template<scalarField F>
    double interpolate(size_t Tdim /*!< Mesh entity dimension */, size_t iT /*!< Entity index */, 
                       int i /*!< Local dof index */, unsigned r /*!<Interpolation degree */) {
      assert(Tdim < 4 && "Entity dimension greater than 3");
      Eigen::Vector3d x0;
      _S1::InterpolatorType Ix, Iy, Iz;
      auto fetchI = [this,&Ix,&Iy,&Iz](int t1, int t2, int t3) {
        const std::array<int,3> iS = splitIndex(i,(t1==0)?_S1.dofV:_S1.dofE,
                                                  (t2==0)?_S2.dofV:_S2.dofE,
                                                  (t3==0)?_S3.dofV:_S3.dofE);
        if (iS[0] < 0) {
          Ix = Iy = Iz = _S1::InterpolatorType{};
        } else {
          Ix = (t1==0)? _S1.IV(iS[0],r) : _S1.IE(iS[0],r);
          Iy = (t2==0)? _S2.IV(iS[1],r) : _S2.IE(iS[1],r);
          Iz = (t3==0)? _S3.IV(iS[2],r) : _S3.IE(iS[2],r);
        }
      };
      if (Tdim == 0) {
        assert(i < dofV && "Local index too large");
        x0 = _mesh.XV(iT);
        fetchI(0,0,0);
      } else if (Tdim == 1) {
        x0 = _mesh.XE(iT);
        const size_t axis = iT/_mesh.IEX;
        assert(i < dofE.at(axis) && "Local index too large");
        if (axis == 0) { // Along x
          fetchI(1,0,0);
        } else if (axis == 1) {
          fetchI(0,1,0);
        } else {
          fetchI(0,0,1);
        }
      } else if (Tdim == 2) {
        x0 = _mesh.XF(iT);
        const size_t axis = iT/_mesh.IFX;
        assert(i < dofF.at(axis) && "Local index too large");
        if (axis == 0) { // Normal along z
          fetchI(1,1,0);
        } else if (axis == 1) {
          fetchI(0,1,1);
        } else {
          fetchI(1,0,1);
        }
      } else {
        x0 = _mesh.XT(iT);
        assert(i < dofT && "Local index too large");
        fetchI(1,1,1);
      }
      double rv = 0;
      for (size_t dx = 0; dx < Ix.size(); ++dx) {
        for (size_t dy = 0; dy < Iy.size(); ++dy) {
          for (size_t dz = 0; dz < Iz.size(); ++dz) {
            for (const &nodeX : Ix[dx]) {
              for (const &nodeY : Iy[dy]) {
                for (const &nodeZ : Iz[dz]) {
                  const Eigen::Vector3d xiqn{nodeX[1],nodeY[1],nodeZ[1]};
                  rv += nodeX[0]*nodeY[0]*nodeZ[0]*f(x0+xiqn,dx,dy,dz);
                }
              }
            }
          }
        }
      }
      return rv;
    }
    /// Evaluate the discrete function uh at x
    double evaluate(const Eigen::Vector3d &x, const Eigen::VectorXd &uh) const {
      const size_t iT = _mesh.findCell(x); // TODO implement this
      const Eigen::Vector3d xloc = x - _mesh.XT(iT);
      double rv = 0.;
      for (size_t Fdim = 0; Fdim < 4; ++Fdim) {
        for (size_t iFT = 0; iFT < _mesh.bDim[3][Fdim]; ++iFT) {
          const size_t iF = _mesh.bN(3,Fdim,iT,iFT);
          const size_t globalOffset = // TODO iF -> offset
          const size_t lDim = Fdim==0?dofV:(Fdim==1?dofE[_mesh.iETtoAxis[iFT]]:(Fdim==2?dofF[_mesh.iFTtoAxis[iFT]]:dofT));
          for (size_t ind = 0; ind < lDim; ++ind) {
            const std::array<int,3> sBasis = splitBasis(Fdim,iFT,ind);
            rv += uh[globalOffset+ind]*_S1.phi(sBasis[0],xloc[0])*_S2.phi(sBasis[1],xloc[1])*_S3.phi(sBasis[2],xloc[2]);
          }
        }
      }
      return rv;
    }
  private:
    const double _h;
    const Spline<_d1,_r1> _S1;
    const Spline<_d2,_r2> _S2;
    const Spline<_d3,_r3> _S3;
    const Mesh _mesh;
}

#endif
