#ifndef TENSORSPLINE_HPP
#define TENSORSPLINE_HPP

#include "spline.hpp"
#include "mesh.hpp"

#include <Eigen/Sparse>
#include <forward_list>

/// Type of the function to interpolate. The suitable component must be extracted. The inputs dx,dy and dz
template<typename T> concept scalarField = requires(T f, const Eigen::Vector3d &x, unsigned dx, unsigned dy, unsigned dz) {{f(x,dx,dy,dz)} -> std::convertible_to<double>;};

/// Class for the tensor product of three splines
template<std::array<int,6> _dr>
class TensorSpline {
  private:
    using S1T = Spline<_dr[0],_dr[1]>; using S2T = Spline<_dr[2],_dr[3]>; using S3T = Spline<_dr[4],_dr[5]>;
    const double _h;
    S1T _S1;
    S2T _S2;
    S3T _S3;
    const Mesh _mesh;
  public:
    TensorSpline(size_t n) : _h(1./n), _S1(_h), _S2(_h), _S3(_h), _mesh(n),
      gOffsetEx(_mesh.nbC[0]*dofV),
      gOffsetEy(gOffsetEx + _mesh.IEX*dofE[0]),
      gOffsetEz(gOffsetEy + _mesh.IEX*dofE[1]),
      gOffsetFz(gOffsetEz + _mesh.IEX*dofE[2]),
      gOffsetFx(gOffsetFz + _mesh.IFX*dofF[0]),
      gOffsetFy(gOffsetFx + _mesh.IFX*dofF[1]),
      gOffsetT(gOffsetFy + _mesh.IFX*dofF[2]),
      nbDofs(gOffsetT + _mesh.nbC[3]*dofT)
    {}
    const Mesh& mesh() const {return _mesh;}
    template<unsigned i> constexpr auto &Si() const {
      if constexpr (i == 0) {
        return _S1;
      } else if constexpr (i == 1) {
        return _S2;
      } else {
        static_assert(i == 2);
        return _S3;
      }
    }
    static constexpr size_t tensorDof(const std::array<unsigned,3> &dLoc,unsigned l = 0, unsigned r = 3) {
      assert(l <= r && r < 4 && dLoc[0] < 2 && dLoc[1] < 2 && dLoc[2] < 2);
      size_t rv = 1;
      switch(l) {
        case 0:
          if (r < 1) break;
          rv *= S1T::dof[dLoc[0]];
        case 1:
          if (r < 2) break;
          rv *= S2T::dof[dLoc[1]];
        case 2:
          if (r < 3) break;
          rv *= S3T::dof[dLoc[2]];
        default:
          ;
      }
      return rv;
    }
    /// Repartition of dof : 0 = Vertex, 1 = Edge;
    static constexpr std::array<unsigned,3> dofVr = {0,0,0};
    static constexpr std::array<std::array<unsigned,3>,3> dofEr = {{{1,0,0},{0,1,0},{0,0,1}}}; // Along x, y, then z
    static constexpr std::array<std::array<unsigned,3>,3> dofFr = {{{1,1,0},{0,1,1},{1,0,1}}}; // Along z, x, then y

    static constexpr std::array<unsigned,3> dofTr = {1,1,1};
    /// Number of degree of freedoms associated to each element. 
    /**
      \warning The number associated to edges and faces varies depending on the orientation
    */
    static constexpr size_t dofV = tensorDof(dofVr);
    static constexpr std::array<size_t,3> dofE = {tensorDof(dofEr[0]),tensorDof(dofEr[1]),tensorDof(dofEr[2])};
    static constexpr std::array<size_t,3> dofF = {tensorDof(dofFr[0]),tensorDof(dofFr[1]),tensorDof(dofFr[2])};
    static constexpr size_t dofT = tensorDof(dofTr);
    /// Global offsets of the corresponding elements (in the spline space)
    const size_t gOffsetEx, gOffsetEy, gOffsetEz, gOffsetFz, gOffsetFx, gOffsetFy, gOffsetT, nbDofs;
    std::array<unsigned,3> localDof(size_t dim, size_t iT) const {
      if (dim == 0) {
        return dofVr;
      } else if (dim == 1) {
        return dofEr[iT/_mesh.IEX];
      } else if (dim == 2) {
        return dofFr[iT/_mesh.IFX];
      } else {
        return dofTr;
      }
    }
    size_t localDim(size_t dim, size_t iT) const {return tensorDof(localDof(dim,iT));}
    size_t gOffset(size_t dim, size_t iT) const {
      switch(dim) {
        case 0:
          return iT*dofV;
        case 1:
          switch(iT/_mesh.IEX) {
            case 0:
              return gOffsetEx + (iT%_mesh.IEX)*dofE[iT/_mesh.IEX];
            case 1:
              return gOffsetEy + (iT%_mesh.IEX)*dofE[iT/_mesh.IEX];
            default:
              return gOffsetEz + (iT%_mesh.IEX)*dofE[iT/_mesh.IEX];
          }
        case 2:
          switch(iT/_mesh.IFX) {
            case 0:
              return gOffsetFz + (iT%_mesh.IFX)*dofF[iT/_mesh.IFX];
            case 1:
              return gOffsetFx + (iT%_mesh.IFX)*dofF[iT/_mesh.IFX];
            default:
              return gOffsetFy + (iT%_mesh.IFX)*dofF[iT/_mesh.IFX];
          }
        default:
          return gOffsetT + iT*dofT;
      }
    }
    /// Retrieve the dimension, element, and local dof from a global one
    std::array<size_t,3> globalToLocal(size_t gOff /*!< Global unknown */) const {
      using T = std::array<size_t,3>;
      assert(gOff < nbDofs);
      if (gOff < gOffsetEx) {
        const size_t dOff = gOff - 0;
        return T{0,dOff/dofV,dOff%dofV};
      } else if (gOff < gOffsetEy) {
        const size_t dOff = gOff - gOffsetEx;
        return T{1,dOff/dofE[0],dOff%dofE[0]};
      } else if (gOff < gOffsetEz) {
        const size_t dOff = gOff - gOffsetEy;
        return T{1,dOff/dofE[1]+_mesh.IEX,dOff%dofE[1]};
      } else if (gOff < gOffsetFz) {
        const size_t dOff = gOff - gOffsetEz;
        return T{1,dOff/dofE[2]+2*_mesh.IEX,dOff%dofE[2]};
      } else if (gOff < gOffsetFx) {
        const size_t dOff = gOff - gOffsetFz;
        return T{2,dOff/dofF[0],dOff%dofF[0]};
      } else if (gOff < gOffsetFy) {
        const size_t dOff = gOff - gOffsetFx;
        return T{2,dOff/dofF[1]+_mesh.IFX,dOff%dofF[1]};
      } else if (gOff < gOffsetT) {
        const size_t dOff = gOff - gOffsetFy;
        return T{2,dOff/dofF[2]+2*_mesh.IFX,dOff%dofF[2]};
      } else {
        const size_t dOff = gOff - gOffsetT;
        return T{3,dOff/dofT,dOff%dofT};
      }
    }
    /// Image space of the derivative
    template<unsigned direction>
    static constexpr std::array<int,6> derivativeSpace() {
      static_assert(direction < 3 && "Use direction = 0 for x, 1 for y, and 2 for z");
      if constexpr (direction == 0) {
        return {_dr[0]-1,_dr[1]-1,_dr[2],_dr[3],_dr[4],_dr[5]};
      } else if constexpr (direction == 1) {
        return {_dr[0],_dr[1],_dr[2]-1,_dr[3]-1,_dr[4],_dr[5]};
      } else {
        return {_dr[0],_dr[1],_dr[2],_dr[3],_dr[4]-1,_dr[5]-1};
      }
    }
    template<unsigned direction>
    Eigen::SparseMatrix<double> derivative() const {
      static_assert(direction < 3 && "Use direction = 0 for x, 1 for y, and 2 for z");
      TensorSpline<derivativeSpace<direction>()> Sp1(_mesh.Nx);
      Eigen::SparseMatrix<double> rv(Sp1.nbDofs,nbDofs);
      std::forward_list<Eigen::Triplet<double>> triplets;
      auto tensorID = [&triplets]<std::array<unsigned,3> dLoc,int dofR, int dofC>(
          const Eigen::Matrix<double,dofR,dofC>& diff, 
          int offsetR, int offsetC){
        constexpr int rep = tensorDof(dLoc,0,direction);
        constexpr int length = tensorDof(dLoc,direction+1,3);
        constexpr int strideR = length*dofR, strideC = length*dofC;
        for (int iInner = 0; iInner < rep; ++iInner) { // Iterate tensor dimensions on the left
          const int baseR = offsetR+iInner*strideR, baseC = offsetC+iInner*strideC;
          for (int iDest = 0; iDest < dofR; ++iDest) { // Iterate tensor dimension corresponding to direction
            for (int iSource = 0; iSource < dofC; ++iSource) { // Iterate tensor dimension corresponding to direction
              const double val = diff(iDest,iSource);
              for (int iOuter = 0; iOuter < length; ++iOuter) { // Iterate tensor dimensions on the right
                triplets.emplace_front(baseR+iDest*length+iOuter,baseC+iSource*length+iOuter,val);
              }
            }
          }
        }
      };
      auto iterateObjects = [this,&Sp1,tensorID]<size_t iDim,unsigned axis>() {
        constexpr std::array<unsigned,3> dLoc = (iDim == 0)? dofVr : (
                                                (iDim == 1)? dofEr[axis] : (
                                                (iDim == 2)? dofFr[axis] : dofTr));
        constexpr std::array<int,2> dofip1 = Sp1.template Si<direction>().dof;
        // Cannot use this in constexpr, but decltype does not evaluate the function
        constexpr std::array<int,2> dofi = std::remove_cvref_t<decltype(Si<direction>())>::dof; 
        if (Sp1.tensorDof(dLoc) == 0) return;
        const size_t maxIT = (iDim == 1)? _mesh.IEX : ((iDim == 2)? _mesh.IFX : _mesh.nbC[iDim]);
        const size_t offsetIT = (iDim == 1)? axis*_mesh.IEX : ((iDim == 2)? axis*_mesh.IFX : 0);
        for (size_t iT0 = 0; iT0 < maxIT; ++iT0) {
          const size_t iT = iT0 + offsetIT;
          const size_t offsetR = Sp1.gOffset(iDim,iT), offsetC = gOffset(iDim,iT);
          if (dLoc[direction] == 0) { // Vertex dof in the derivative direction
            constexpr int dofR = dofip1[0], dofC = dofi[0];
            tensorID.template operator()<dLoc,dofR,dofC>(Si<direction>().dx.template topLeftCorner<dofR,dofC>(),offsetR,offsetC); // V->V component
          } else { // Edge dof in the derivative direction
            constexpr int dofR = dofip1[1], dofCV = dofi[0], dofCE = dofi[1];
            tensorID.template operator()<dLoc,dofR,dofCE>(Si<direction>().dx.template bottomRightCorner<dofR,dofCE>(),offsetR,offsetC); // E->E component
            constexpr size_t VDim = iDim - 1; // Dimension of the "vertices" of E
            static_assert(dLoc[direction] == 0 || VDim == dLoc[(direction+1)%3]+dLoc[(direction+2)%3]); 
            auto getLRFace = [this](size_t iT) { // If VDim = 2, the element must be a cell
              constexpr size_t iFT = []<typename M>(){
                for (size_t i = 0; i < 6; ++i) {
                  if (M::iFTtoAxis[i] == ((direction+1)%3)) return i;
                } // iFT is the first face normal to direction
              }.template operator()<decltype(_mesh)>();
              static_assert(iFT < 5);
              return std::array<size_t,2>{_mesh.bT(iT,iFT),_mesh.bT(iT,iFT+1)}; // Faces are ordered with increasing global index
            };
            auto getLREdge = [this](size_t iF) { // If VDim = 1, the element must be a face spawned by direction and any of the edges of interest
              constexpr size_t iEF = []<typename M>(){
                for (size_t i = 0; i < 4; ++i) {
                  if (M::iEFtoAxis[axis][i] != direction) return i;
                } // iEF is the first edge orthogonal to direction
              }.template operator()<decltype(_mesh)>();
              static_assert(iEF < 3);
              return std::array<size_t,2>{_mesh.bF(iF,iEF),_mesh.bF(iF,iEF+1)}; // Edges are ordered with increasing global index
            };
            const auto [iV0, iV1] = (VDim == 2)? getLRFace(iT) : (
                                    (VDim == 1)? getLREdge(iT) : 
                                    std::array<size_t,2>{_mesh.bE(iT,0),_mesh.bE(iT,1)});
            tensorID.template operator()<dLoc,dofR,dofCV>(Si<direction>().dx.template bottomLeftCorner<dofR,dofCV>(),offsetR,gOffset(VDim,iV0));
            tensorID.template operator()<dLoc,dofR,dofCV>(Si<direction>().dx.template block<dofR,dofCV>(2*dofip1[0],dofCV),offsetR,gOffset(VDim,iV1));
          }
        }
      };
      iterateObjects.template operator()<0,0>(); // Vertices
      iterateObjects.template operator()<1,0>(); // Edges
      iterateObjects.template operator()<1,1>();
      iterateObjects.template operator()<1,2>();
      iterateObjects.template operator()<2,0>(); // Faces
      iterateObjects.template operator()<2,1>();
      iterateObjects.template operator()<2,2>();
      iterateObjects.template operator()<3,0>(); // Cells
      rv.setFromTriplets(triplets.begin(),triplets.end());
      return rv;
    }
    /// Convert the local index of the tensor product to the locals indices of each components
    static constexpr std::array<int,3> splitIndex(int i,const std::array<unsigned,3> &dLoc) {
      assert(dLoc[0] < 2 && dLoc[1] < 2 && dLoc[2] < 2);
      const int S1 = S1T::dof[dLoc[0]], S2 = S2T::dof[dLoc[1]], S3 = S3T::dof[dLoc[2]];
      if (S1*S2*S3 == 0) return {-1,-1,-1};
      const int i2 = i%(S2*S3), ix = i/(S2*S3);
      const int iz = i2%S3, iy = i2/S3;
      return {ix,iy,iz};
    }
    /// Interpolate a scalar field
    template<scalarField F>
    double interpolate(F f, size_t Tdim /*!< Mesh entity dimension */, size_t iT /*!< Entity index */, 
                       size_t i /*!< Local dof index */, int r /*!<Interpolation degree */) {
      assert(Tdim < 4 && "Entity dimension greater than 3");
      Eigen::Vector3d x0;
      typename S1T::InterpolatorType Ix, Iy, Iz;
      auto fetchI = [this,&Ix,&Iy,&Iz,i,r](const std::array<unsigned,3> &dLoc) {
        const std::array<int,3> iS = splitIndex(i,dLoc);
        if (iS[0] < 0) {
          Ix = Iy = Iz = typename S1T::InterpolatorType{};
        } else {
          Ix = (dLoc[0]==0)? _S1.IV(iS[0],(r<0)?2*S1T::degree+1:r) : _S1.IE(iS[0],(r<0)?2*S1T::degree+1:r);
          Iy = (dLoc[1]==0)? _S2.IV(iS[1],(r<0)?2*S2T::degree+1:r) : _S2.IE(iS[1],(r<0)?2*S2T::degree+1:r);
          Iz = (dLoc[2]==0)? _S3.IV(iS[2],(r<0)?2*S3T::degree+1:r) : _S3.IE(iS[2],(r<0)?2*S3T::degree+1:r);
        }
      };
      if (Tdim == 0) {
        assert(i < dofV && "Local index too large");
        x0 = _mesh.XV(iT);
        fetchI(dofVr);
      } else if (Tdim == 1) {
        x0 = _mesh.XE(iT);
        const size_t axis = iT/_mesh.IEX;
        assert(i < dofE.at(axis) && "Local index too large");
        fetchI(dofEr[axis]);
      } else if (Tdim == 2) {
        x0 = _mesh.XF(iT);
        const size_t axis = iT/_mesh.IFX;
        assert(i < dofF.at(axis) && "Local index too large");
        fetchI(dofFr[axis]);
      } else {
        x0 = _mesh.XT(iT);
        assert(i < dofT && "Local index too large");
        fetchI(dofTr);
      }
      double rv = 0;
      for (size_t dx = 0; dx < Ix.size(); ++dx) {
        for (size_t dy = 0; dy < Iy.size(); ++dy) {
          for (size_t dz = 0; dz < Iz.size(); ++dz) {
            for (const auto &nodeX : Ix[dx]) {
              for (const auto &nodeY : Iy[dy]) {
                for (const auto &nodeZ : Iz[dz]) {
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
    /// Convert the local index on a cell to the local index in the splines components
    std::array<int,3> splitBasis(size_t Fdim /*!< Entity dimension */,
        size_t iFT /*!< Local index of the entity within the cell */, size_t ind /*!< Local index of the dof associated with iFT */) const {
      assert(Fdim < 4);
      assert(ind >= 0);
      assert(iFT < _mesh.bDim[3][Fdim]);
      if (Fdim == 0) {
        assert(ind < dofV);
        std::array<int,3> splitI = splitIndex(ind,dofVr);
        assert(splitI[0] >= 0);
        splitI[0] += _mesh.iVTtoSide[iFT][0]*S1T::dofV;
        splitI[1] += _mesh.iVTtoSide[iFT][1]*S2T::dofV;
        splitI[2] += _mesh.iVTtoSide[iFT][2]*S3T::dofV;
        return splitI;
      } else if (Fdim == 1) {
        const size_t Axis = _mesh.iETtoAxis[iFT];
        assert(ind < dofE[Axis]);
        std::array<int,3> splitI = splitIndex(ind,dofEr[Axis]);
        assert(splitI[0] >= 0);
        splitI[0] += (_mesh.iETtoSide[iFT][0]+2*dofEr[Axis][0])*S1T::dofV;
        splitI[1] += (_mesh.iETtoSide[iFT][1]+2*dofEr[Axis][1])*S2T::dofV;
        splitI[2] += (_mesh.iETtoSide[iFT][2]+2*dofEr[Axis][2])*S3T::dofV;
        return splitI;
      } else if (Fdim == 2) {
        const size_t Axis = _mesh.iFTtoAxis[iFT];
        assert(ind < dofF[Axis]);
        std::array<int,3> splitI = splitIndex(ind,dofFr[Axis]);
        assert(splitI[0] >= 0);
        splitI[0] += (_mesh.iFTtoSide[iFT][0]+2*dofFr[Axis][0])*S1T::dofV;
        splitI[1] += (_mesh.iFTtoSide[iFT][1]+2*dofFr[Axis][1])*S2T::dofV;
        splitI[2] += (_mesh.iFTtoSide[iFT][2]+2*dofFr[Axis][2])*S3T::dofV;
        return splitI;
      } else {
        assert(Fdim == 3);
        assert(ind < dofT);
        std::array<int,3> splitI = splitIndex(ind,dofTr);
        assert(splitI[0] >= 0);
        splitI[0] += 2*S1T::dofV;
        splitI[1] += 2*S2T::dofV;
        splitI[2] += 2*S3T::dofV;
        return splitI;
      }
    }
    /// Evaluate the discrete function uh at x
    double evaluate(const Eigen::Vector3d &x, const Eigen::Ref<const Eigen::VectorXd> &uh) const {
      const size_t iT = _mesh.findCell(x); 
      const Eigen::Vector3d xloc = x - _mesh.XT(iT);
      double rv = 0.;
      for (size_t Fdim = 0; Fdim < 4; ++Fdim) {
        for (size_t iFT = 0; iFT < _mesh.bDim[3][Fdim]; ++iFT) {
          const size_t iF = _mesh.bN(3,Fdim,iT,iFT);
          const size_t globalOffset = gOffset(Fdim,iF);
          const size_t lDim = localDim(Fdim,iF);
          for (size_t ind = 0; ind < lDim; ++ind) {
            const std::array<int,3> sBasis = splitBasis(Fdim,iFT,ind);
            rv += uh[globalOffset+ind]*_S1.phi(sBasis[0],xloc[0])*_S2.phi(sBasis[1],xloc[1])*_S3.phi(sBasis[2],xloc[2]);
          }
        }
      }
      return rv;
    }
    /// Compute the L2 mass matrix
    Eigen::SparseMatrix<double> L2() const {
      Eigen::SparseMatrix<double> rv(nbDofs,nbDofs);
      std::forward_list<Eigen::Triplet<double>> triplets;
      for (size_t iT = 0; iT < _mesh.nbC[3]; ++iT) {
        for (size_t FDim1 = 0; FDim1 < 4; ++FDim1) {
          for (size_t iFT1 = 0; iFT1 < _mesh.bDim[3][FDim1]; ++iFT1) {
            const size_t iF1 = _mesh.bN(3,FDim1,iT,iFT1);
            const size_t offset1 = gOffset(FDim1,iF1);
            const size_t lDim1 = localDim(FDim1,iF1);
            for (size_t FDim2 = FDim1; FDim2 < 4; ++FDim2) {
              for (size_t iFT2 = (FDim2 == FDim1)?iFT1:0; iFT2 < _mesh.bDim[3][FDim2]; ++iFT2) {
                const size_t iF2 = _mesh.bN(3,FDim2,iT,iFT2);
                const size_t offset2 = gOffset(FDim2,iF2);
                const size_t lDim2 = localDim(FDim2,iF2);
                for (size_t ind1 = 0; ind1 < lDim1; ++ind1) {
                  const std::array<int,3> sBasis1 = splitBasis(FDim1,iFT1,ind1);
                  for (size_t ind2 = (FDim2==FDim1&&iFT2==iFT1)?ind1:0; ind2 < lDim2; ++ind2) {
                    const std::array<int,3> sBasis2 = splitBasis(FDim2,iFT2,ind2);
                    const double val = _S1.mass()(sBasis1[0],sBasis2[0])*
                                       _S2.mass()(sBasis1[1],sBasis2[1])*
                                       _S3.mass()(sBasis1[2],sBasis2[2]);
                    triplets.emplace_front(offset1+ind1,offset2+ind2,val);
                    if (offset2+ind2 != offset1+ind1) {
                      triplets.emplace_front(offset2+ind2,offset1+ind1,val);
                    }
                  }
                }
              }
            }
          }
        }
      }
      rv.setFromTriplets(triplets.begin(),triplets.end());
      return rv;
    }
};

#endif
