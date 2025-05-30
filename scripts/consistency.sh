#!/usr/bin/env bash

for ((form = 0; form < 3; form++)); do
  for ((case = 2; case < 4; ++case)); do
    for ((mesh = 1; mesh < 6; ++mesh)); do
      let N=4*$mesh
      ./tests/cc_consistency $form $N 1 $case
    done
  done
done

