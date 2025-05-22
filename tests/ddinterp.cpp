#include <iostream>

#include "../src/divdiv.hpp"

#include "function.hpp"

int main() {
  DivDivSpace<0> v0(4);
  std::cout<<"Space dim: "<<v0.nbDofs<<std::endl;
  return 0;
}
