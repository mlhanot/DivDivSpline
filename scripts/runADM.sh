#!/usr/bin/env bash

for ((case = 0; case < 2; ++case)); do
  for ((dtE = 2; dtE < 5; ++dtE)); do
    for ((mesh = 1; mesh < 6; ++mesh)); do
      let N=2*$mesh
      dt="1e-$dtE"
      outfile="adm_${case}_dt${dtE}_N$N"
      if [ ! -f $outfile ]; then
        ./Schemes/adm $case $N $dt $outfile "-ksp_rtol 1e-7 -ksp_type ibcgs"
      fi
    done
  done
done

