ADD_EXECUTABLE(db_interp interp.cpp)
TARGET_LINK_LIBRARIES(db_interp quadrature)
ADD_EXECUTABLE(db_derivative derivative.cpp)
TARGET_LINK_LIBRARIES(db_derivative quadrature)
ADD_EXECUTABLE(db_mass mass.cpp)
TARGET_LINK_LIBRARIES(db_mass quadrature)
ADD_EXECUTABLE(dd_interp ddinterp.cpp)
TARGET_LINK_LIBRARIES(dd_interp divdiv)
ADD_EXECUTABLE(dd_mass ddmass.cpp)
TARGET_LINK_LIBRARIES(dd_mass divdiv)
ADD_EXECUTABLE(dd_derivative ddderivative.cpp)
TARGET_LINK_LIBRARIES(dd_derivative divdiv)
ADD_EXECUTABLE(dd_complex ddcomplex.cpp)
TARGET_LINK_LIBRARIES(dd_complex divdiv)

#------------------------------------------------------------------------------#
#                             Eigenvalues
#------------------------------------------------------------------------------#

INCLUDE(FindPkgConfig)
FIND_PACKAGE(MPI REQUIRED)
INCLUDE_DIRECTORIES(${MPI_INCLUDE_PATH})
PKG_CHECK_MODULES(PETSC REQUIRED IMPORTED_TARGET PETSc)
PKG_CHECK_MODULES(SLEPC REQUIRED IMPORTED_TARGET SLEPc)

ADD_EXECUTABLE(cc_poincare cc_poincare.cpp)
TARGET_LINK_LIBRARIES(cc_poincare divdiv MPI::MPI_CXX PkgConfig::PETSC PkgConfig::SLEPC)
ADD_EXECUTABLE(cc_consistency cc_consistency.cpp)
TARGET_LINK_LIBRARIES(cc_consistency divdiv MPI::MPI_CXX PkgConfig::PETSC)
