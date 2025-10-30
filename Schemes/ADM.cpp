#include "ADM.hpp"

#include "solver.hpp"
#include "function_adj.hpp"
#include "timer.hpp"

#include <iostream>
#include <fstream>

int main(int argc,char **argv) {
  if (argc < 4) {
    std::cout<<"Usage: Case NbCells dt outfile\n";
    return 1;
  }
  const int Case = std::atoi(argv[1]);
  const int N = std::atoi(argv[2]);
  const double dt = std::atof(argv[3]);
  const char *filename = (argc > 4)? argv[4] : nullptr;
  constexpr double TF = 2.*std::acos(-1.);
  if (Case < 0 || Case > 2) {
    std::cerr<<"Case should be 0 or 1. Got "<<Case<<std::endl;
    return 1;
  }
  if (N < 2) {
    std::cerr<<"Number of cell too low: "<<N<<std::endl;
    return 1;
  }
  if (dt < 1e-8) {
    std::cerr<<"Time step too small: "<<dt<<std::endl;
    return 1;
  }

  std::ofstream fh;
  if (filename) {
    fh.open(filename);
  }
  std::cout<<"ADM solver: testCase: "<<Case<<", parameters: Subdivision: "<<N<<", Timestep: "<<dt<<std::endl;
  Timer startTime;
  ADM<true> adm(N,dt);
  startTime.stop();
  std::cout<<"Assembled the system in "<<startTime<<std::endl;
  adm.markAllBoundary();
  union {
    Sol0 s0;
    Sol1 s1;
  } sol;
  auto solTime = [Case,&sol]()->double& {
    if (Case == 0) return sol.s0.t;
    else return sol.s1.t;
  };
  auto interpolate = [Case,&sol,&adm](Eigen::VectorXd &u) {
    if (Case == 0) adm.interpolate(u,sol.s0);
    else adm.interpolate(u,sol.s1);
  };
  const std::array<Eigen::SparseMatrix<double>,4> masses({adm.mass(0),adm.mass(1),adm.mass(2),adm.mass(3)});
  auto splitNorm = [&adm,&masses](const Eigen::VectorXd &a) {
    return std::to_array({std::sqrt((masses[0]*a.head(adm.offset(1))).dot(a.head(adm.offset(1)))),
           std::sqrt((masses[1]*a.segment(adm.offset(1),adm.offset(2)-adm.offset(1))).dot(
             a.segment(adm.offset(1),adm.offset(2)-adm.offset(1)))),
           std::sqrt((masses[2]*a.segment(adm.offset(2),adm.offset(3)-adm.offset(2))).dot(
           a.segment(adm.offset(2),adm.offset(3)-adm.offset(2)))),
           std::sqrt((masses[3]*a.tail(adm.offset(4)-adm.offset(3))).dot(
           a.tail(adm.offset(4)-adm.offset(3))))
           });
  };
  const Eigen::SparseMatrix<double> A = adm.system(), bExt = adm.boundaryExt(), iExt = adm.interiorExt();
  const size_t interiorSize = A.cols(), boundarySize = bExt.cols(), totalSize = interiorSize+boundarySize;
  std::cout<<"System size: "<<interiorSize<<", boundary size: "<<boundarySize<<", total: "<<totalSize<<std::endl;
  // Setup solver
  std::shared_ptr<Solver> solver;
  if (interiorSize < 1.3e5) { // Up to h = 10^-1
    solver = std::make_shared<SolverLU>();
  } else {
    PetscCall(PetscInitialize(&argc,&argv,NULL,NULL));
    solver = std::make_shared<SolverPetsc>();
  }
  Timer setupTime;
  solver->setup(A);
  setupTime.stop();
  std::cout<<"Factorised the system in "<<setupTime<<std::endl;
  // Initialize
  solTime() = 0.;
  Eigen::VectorXd refVal = Eigen::VectorXd::Zero(totalSize);
  interpolate(refVal);
  Eigen::VectorXd uOld = iExt.transpose()*refVal;
  Eigen::VectorXd uBOld = bExt.transpose()*refVal;
  while (solTime() < TF) {
    solTime() += dt;
    Timer interpolateTime;
    interpolate(refVal); 
    interpolateTime.stop();
    const Eigen::VectorXd uB = bExt.transpose()*refVal;
    const Eigen::VectorXd rhs = adm.RHS(uOld,uB,uBOld);
    Timer solveTime;
    const Eigen::VectorXd u = solver->solve(rhs);
    solveTime.stop();
    checkSolution(A,u,rhs);
    // Compute norm and prepare next iteration 
    const Eigen::VectorXd fullU = iExt*u + bExt*uB;
    std::array<double,4> norms = splitNorm(refVal), errors = splitNorm(refVal-fullU);
    uOld = u;
    uBOld = uB;
    // Print results
    std::cout<<"t: "<<solTime()<<" Errors: "<<errors[0]<<", "<<errors[1]<<", "<<errors[2]<<", "<<errors[3]<<"\n";
    std::cout<<"\tNorms: "<<norms[0]<<", "<<norms[1]<<", "<<norms[2]<<", "<<norms[3]<<"\n";
    std::cout<<"\tInterpolation time: "<<interpolateTime<<", solve time: "<<solveTime<<std::endl;
    if (fh) {
      fh<<solTime()<<", "<<errors[0]<<", "<<errors[1]<<", "<<errors[2]<<", "<<errors[3]<<"\n";
    }
  }
  startTime.stop();
  std::cout<<"Tf reached. Total execution time: "<<startTime<<std::endl;
  return 0;
}

