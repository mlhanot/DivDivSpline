#ifndef MESH_INCLUDED
#define MESH_INCLUDED

#include <array>
#include <stddef.h>
#include <cassert>
#include <Eigen/Dense>

/// Compute basic geometrical relation in a cartesian mesh
/**
  The mesh is the cube [0,1]^3
  The vertices are indexed starting from (0,0,0) in increasing order along the x, y, then z axis.
  The edges and faces are split in 3 groups (one for each direction).
  */
struct Mesh {
  Mesh(size_t Nsub) : 
    Nx(Nsub),
    IEX(Nx*(Nx+1)*(Nx+1)), // The edges are oriented along x, y, and then z
    IFX(Nx*Nx*(Nx+1)), // The normal of the faces are along z, x and then y
    offsetX({1,(Nx+1),(Nx+1)*(Nx+1)}),
    offsetXE({{ {1,Nx,Nx*(Nx+1)}, // edge along x
                {1,Nx+1,Nx*(Nx+1)}, // edge along y
                {1,Nx+1,(Nx+1)*(Nx+1)} }}), // edge along z
    nbC({(Nx+1)*(Nx+1)*(Nx+1),3*IEX,3*IFX,Nx*Nx*Nx}),
    hE(1./Nx),
    VF(hE*hE),
    VT(VF*hE) {}
  const size_t Nx, IEX, IFX;
  const std::array<size_t,3> offsetX; // Offset to the next vertex in the direction {x,y,z}
  const std::array<std::array<size_t,3>,3> offsetXE; // Offset to the next edge along {x,y,z} in the direction {x,y,z}
  const std::array<size_t,4> nbC;
  static constexpr std::array<std::array<size_t,4>,4> bDim{{{1,0,0,0},{2,1,0,0},{4,4,1,0},{8,12,6,1}}};
  size_t bE(size_t iE, size_t iVE) const {
    assert(iVE < 2 && iE < nbC[1]);
    const size_t axis = iE/IEX, iEN = iE%IEX;
    // Correct for the discrepancy between edge and vertex indexing
    const size_t iENcorr = iEN + ((axis == 0)? (iEN/Nx) : 
                                 ((axis == 1)? (iEN/(Nx*(Nx+1)))*(Nx+1) : 0));
    switch(iVE) {
      case 0:
        return iENcorr;
      default:
        return iENcorr + offsetX[axis];
    }
  }
  /// Return the global index of the edge iEF of iF
  /**
    The global index of the vertices of iF are given by 
    [bE(bF(iF)[0])[0],bE(bF(iF)[0])[1],bE(bF(iF)[1])[0],bE(bF(iF)[1])[1]]
    */
  size_t bF(size_t iF, size_t iEF) const {
    assert(iEF < 4 && iF < nbC[2]);
    const size_t axis = iF/IFX, iFN = iF%IFX;
    const size_t iex = axis*IEX, iey = ((axis+1)%3)*IEX;
    switch(iEF) {
      case 0:
        return iFN + ((axis == 0)? (iFN/(Nx*Nx))*Nx : ((axis == 1)? 0 : iFN/Nx)) + iex;
      case 1:
        return iFN + ((axis == 0)? (iFN/(Nx*Nx))*Nx : ((axis == 1)? 0 : iFN/Nx)) + iex + offsetXE[axis][(axis+1)%3];
      case 2:
        return iFN + ((axis == 0)? iFN/Nx : ((axis == 1)? (iFN/(Nx*(Nx+1)))*(Nx+1) : 0)) + iey;
      default:
        return iFN + ((axis == 0)? iFN/Nx : ((axis == 1)? (iFN/(Nx*(Nx+1)))*(Nx+1) : 0)) + iey + offsetXE[(axis+1)%3][axis];
    }
  }
  /// Return the global index of the vertex iVF of iF
  size_t bVF(size_t iF, size_t iVF) const {
    assert(iVF < 4 && iF < nbC[2]);
    return bE(bF(iF,iVF/2),iVF%2);
  }
  /// Return the global index of the face iFT of iT
  /**
    Denoting by (i,j) = E(F_i)_j, the 12 edges are given by 
    [(1,1),(1,2),(1,3),(1,4),(2,1),(2,2),(2,3),(2,4),(3,3),(5,2),(6,1),(4,4)]
    [(5,3),(6,3),(3,1),(4,1),(5,4),(6,4),(3,2),(4,2),(5,1),(4,3),(3,4),(6,2)]

    The vertices are given by the vertices of the first two faces
    */
  size_t bT(size_t iT, size_t iFT) const {
    assert(iFT < 6 && iT < nbC[3]);
    switch(iFT) {
      case 0:
        return iT + 0*IFX;
      case 1:
        return iT + 0*IFX + Nx*Nx;
      case 2:
        return iT + iT/Nx + 1*IFX;
      case 3:
        return iT + iT/Nx + 1*IFX + 1;
      case 4:
        return iT + (iT/(Nx*Nx))*Nx + 2*IFX;
      default:
        return iT + (iT/(Nx*Nx))*Nx + 2*IFX + Nx;
    }
  }
  /// Return the global index of the edge iET of iT
  size_t bET(size_t iT, size_t iET) const {
    assert(iET < 12 && iT < nbC[3]);
    if (iET < 4) {
      return bF(bT(iT,0),iET);
    } else if (iET < 8) {
      return bF(bT(iT,1),iET-4);
    } else if (iET == 8) {
      return bF(bT(iT,2),2);
    } else if (iET == 9) {
      return bF(bT(iT,4),1);
    } else if (iET == 10) {
      return bF(bT(iT,5),0);
    } else {
      return bF(bT(iT,3),3);
    }
  }
  /// Return the global index of the vertex iVT of iT
  size_t bVT(size_t iT, size_t iVT) const {
    assert(iVT < 8 && iT < nbC[3]);
    if (iVT < 4) {
      return bVF(bT(iT,0),iVT);
    } else {
      return bVF(bT(iT,1),iVT-4);
    }
  }
  /// Return the global index of the locInd-th bDim dimensional element of the ind-th dim dimensional element
  /** 
    Added to allow runtime dispatch in loop based generic code
    */
  size_t bN(size_t dim, size_t bDim, size_t ind, size_t locInd) const {
    assert(dim < 4 && bDim <= dim);
    switch(dim) {
      case(3):
        switch(bDim) {
          case(2):
            return bT(ind,locInd);
          case(1):
            return bET(ind,locInd);
          case(0):
            return bVT(ind,locInd);
          default:
            return ind; // Assumes dDim == dim
        }
      case(2):
        switch(bDim) {
          case(1):
            return bF(ind,locInd);
          case(0):
            return bVF(ind,locInd);
          default:
            return ind;
        }
      case(1):
        switch(bDim) {
          case(0):
            return bE(ind,locInd);
          default:
            return ind;
        }
      default:
        return ind;
    }
  }
  /// Return the reverse mapping from local vertices of the iEF-th edge of iF to the local vertices of iF
  static constexpr std::array<std::array<size_t,2>,4> ViEFtoiF {{
      std::array<size_t,2>{0,1},
      std::array<size_t,2>{2,3},
      std::array<size_t,2>{0,2},
      std::array<size_t,2>{1,3}
    }};
  /// Return the reverse mapping from local edges of the iFT-th face of iT to the local edges of iT
  static constexpr std::array<std::array<size_t,4>,6> EiFTtoiT {{
      std::array<size_t,4>{0,1,2,3},
      std::array<size_t,4>{4,5,6,7},
      std::array<size_t,4>{2,6,8,10},
      std::array<size_t,4>{3,7,9,11},
      std::array<size_t,4>{8,9,0,4},
      std::array<size_t,4>{10,11,1,5}
    }};
  /// Return the reverse mapping from local vertices of the iFT-th face of iT to the local vertices of iT
  static constexpr std::array<std::array<size_t,4>,6> ViFTtoiT {{
      std::array<size_t,4>{0,1,2,3},
      std::array<size_t,4>{4,5,6,7},
      std::array<size_t,4>{0,2,4,6},
      std::array<size_t,4>{1,3,5,7},
      std::array<size_t,4>{0,4,1,5},
      std::array<size_t,4>{2,6,3,7}
    }};
  /// Return the reverse mapping from local vertices of the iET-th edge of iT to the local vertices of iT
  static constexpr std::array<std::array<size_t,2>,12> ViETtoiT {{
      std::array<size_t,2>{0,1},
      std::array<size_t,2>{2,3},
      std::array<size_t,2>{0,2},
      std::array<size_t,2>{1,3},
      std::array<size_t,2>{4,5},
      std::array<size_t,2>{6,7},
      std::array<size_t,2>{4,6},
      std::array<size_t,2>{5,7},
      std::array<size_t,2>{0,4},
      std::array<size_t,2>{1,5},
      std::array<size_t,2>{2,6},
      std::array<size_t,2>{3,7}
    }};
  static constexpr std::array<double,2> wEV{-1,1};
  static constexpr std::array<double,4> wFE{-1,1,1,-1};
  static constexpr std::array<double,6> wTF{-1,1,-1,1,-1,1};
  /// Mapping from the local vertices of a cell to the local ordering in the tensor product (0 if the vertex is a start of the standard edge, 1 otherwise)
  static constexpr std::array<std::array<size_t,3>,8> iVTtoSide {{
    std::array<size_t,3>{0,0,0},
    std::array<size_t,3>{1,0,0},
    std::array<size_t,3>{0,1,0},
    std::array<size_t,3>{1,1,0},
    std::array<size_t,3>{0,0,1},
    std::array<size_t,3>{1,0,1},
    std::array<size_t,3>{0,1,1},
    std::array<size_t,3>{1,1,1}
  }};
  /// Mapping from the local edges of a cell to the local ordering in the tensor product (0 if this correspond to the edge component or if the vertex is a start of the standard edge, 1 otherwise)
  static constexpr std::array<std::array<size_t,3>,12> iETtoSide {{
    std::array<size_t,3>{0,0,0},
    std::array<size_t,3>{0,1,0},
    std::array<size_t,3>{0,0,0},
    std::array<size_t,3>{1,0,0},
    std::array<size_t,3>{0,0,1}, // 5
    std::array<size_t,3>{0,1,1},
    std::array<size_t,3>{0,0,1},
    std::array<size_t,3>{1,0,1},
    std::array<size_t,3>{0,0,0}, // 9
    std::array<size_t,3>{1,0,0},
    std::array<size_t,3>{0,1,0},
    std::array<size_t,3>{1,1,0}
  }};
  /// Mapping from the local faces of a cell to the local ordering in the tensor product (0 if this correspond to the edge component or if the vertex is a start of the standard edge, 1 otherwise)
  static constexpr std::array<std::array<size_t,3>,6> iFTtoSide {{
    std::array<size_t,3>{0,0,0},
    std::array<size_t,3>{0,0,1},
    std::array<size_t,3>{0,0,0},
    std::array<size_t,3>{1,0,0},
    std::array<size_t,3>{0,0,0},
    std::array<size_t,3>{0,1,0}
  }};
  /// Retrieve the axis of the iEF-th edge of a face of given axis
  static constexpr std::array<std::array<size_t,4>,3> iEFtoAxis {{
    std::array<size_t,4>{0,0,1,1},
    std::array<size_t,4>{1,1,2,2},
    std::array<size_t,4>{2,2,0,0}
  }};
  /// Retrieve the axis of the iET-th edge of a cell
  static constexpr std::array<size_t,12> iETtoAxis{0,0,1,1,0,0,1,1,2,2,2,2};
  /// Retrieve the axis of the iFT-th face of a cell
  static constexpr std::array<size_t,6> iFTtoAxis{0,0,1,1,2,2};
  const double hE, VF, VT;
  Eigen::Vector3d XV(size_t iV) const {
    assert(iV < nbC[0]);
    return Eigen::Vector3d{hE*(iV%(Nx+1)),hE*((iV/(Nx+1))%(Nx+1)),hE*(iV/((Nx+1)*(Nx+1)))};
  }
  /// Return the point such that E = XE + [0,h]*tE
  Eigen::Vector3d XE(size_t iE) const {
    assert(iE < nbC[1]);
    return XV(bE(iE,0));
  }
  /// Return the point such that F = XF + [0,h]*tF1 + [0,h]*tF2
  Eigen::Vector3d XF(size_t iF) const {
    assert(iF < nbC[2]);
    return XV(bVF(iF,0));
  }
  /// Return the point such that T = XT + [0,h]*x1 + [0,h]*x2 + [0,h]*x3
  Eigen::Vector3d XT(size_t iT) const {
    assert(iT < nbC[3]);
    return XV(bVT(iT,0));
  }
  size_t findCell(const Eigen::Vector3d &x) const {
    const int ix = x(0)*Nx, iy = x(1)*Nx, iz = x(2)*Nx;
    assert(ix >= 0 && iy >= 0 && iz >= 0 && ix < static_cast<int>(Nx) && iy < static_cast<int>(Nx) && iz < static_cast<int>(Nx));
    return ix + Nx*iy + Nx*Nx*iz;
  }
};

#endif //MESH_INCLUDED
