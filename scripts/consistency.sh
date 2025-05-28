#!/usr/bin/env bash

for ((form = 0; form < 3; form++)); do
  for ((case = 0; case < 3; ++case)); do
    for ((mesh = 2; mesh < 6; ++mesh)); do
      let N=2**$mesh
      ./tests/cc_consistency $form $N 1 $case
    done
  done
done

