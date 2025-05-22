project(DivDivSpline)
CMAKE_MINIMUM_REQUIRED(VERSION 3.6)

OPTION(BUILD_TEST "Build the testing tree." ON)

#------------------------------------------------------------------------------#
#                             Compiler setup
#------------------------------------------------------------------------------#
find_program(CCACHE_FOUND ccache)
if(CCACHE_FOUND)
    set_property(GLOBAL PROPERTY RULE_LAUNCH_COMPILE ccache)
    MESSAGE(STATUS "Found ccache")
endif()
SET(CMAKE_CXX_COMPILER "g++")
SET(CMAKE_CXX_STD "-std=c++20")
SET(CMAKE_CXX_FLAGS "-g -Wall -Wpedantic -Werror -Wno-uninitialized")
SET(CMAKE_CXX_FLAGS_DEBUGALL "${CMAKE_CXX_STD} ${CMAKE_CXX_FLAGS} -Og -DEIGEN_INITIALIZE_MATRICES_BY_NAN -fsanitize=address,undefined -pthread" )
SET(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_STD} ${CMAKE_CXX_FLAGS} -Og -DOPTIMIZE -pthread -DEIGEN_INITIALIZE_MATRICES_BY_NAN" ${ADDITIONAL_FLAGS} )
SET(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_STD} ${CMAKE_CXX_FLAGS} -O3 -DNDEBUG -DOPTIMIZE -pthread" ${ADDITIONAL_FLAGS} )
SET(CMAKE_CXX_FLAGS_RELWITHDEBINFO "${CMAKE_CXX_STD} ${CMAKE_CXX_FLAGS} -O3 -g -DNDEBUG -DOPTIMIZE -pthread" ${ADDITIONAL_FLAGS} )

SET(CMAKE_BUILD_TYPE Debug)
#SET(CMAKE_BUILD_TYPE Debugall)
IF(NOT CMAKE_BUILD_TYPE)
  SET(CMAKE_BUILD_TYPE Release CACHE STRING "Choose the type of build: Debug Release RelWithDebInfo" FORCE)
ENDIF(NOT CMAKE_BUILD_TYPE)
MESSAGE(STATUS "Build type : ${CMAKE_BUILD_TYPE}")

#------------------------------------------------------------------------------#
#                             Required libraries
#------------------------------------------------------------------------------#

FIND_PACKAGE(Eigen3 3.4 REQUIRED)
SET(EXTERNAL_INCLUDES ${EXTERNAL_INCLUDES} ${EIGEN3_INCLUDE_DIR})

INCLUDE_DIRECTORIES(${EXTERNAL_INCLUDES})

#------------------------------------------------------------------------------#
#                             Compiled libraries
#------------------------------------------------------------------------------#

ADD_LIBRARY(quadrature src/legendregauss.cpp)
ADD_LIBRARY(divdiv src/divdiv.cpp)
TARGET_LINK_LIBRARIES(divdiv quadrature)

#------------------------------------------------------------------------------#
#                             Tests
#------------------------------------------------------------------------------#

IF(BUILD_TEST)
  ADD_SUBDIRECTORY(tests)
ENDIF()

