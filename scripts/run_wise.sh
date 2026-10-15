#!/bin/bash
# WiSE-FT sweep for seed 0: waits for the trained checkpoint, then exports val/test/xsrc_aof logits per mixing alpha.
cd ~/tw; export PYTHONPATH=src; M=~/data/manifests/dfire_train.csv; R=~/data/dfire
until [ -f ~/runs/lora_s0/xsrc_aof.npz ]; do sleep 60; done
for a in 0.0 0.25 0.5 0.75 1.0; do
  mkdir -p ~/runs/lora_s0/wise_$a
  for p in val test xsrc_aof; do python3 -m fire_assessor.train export --manifest $M --root $R --weights ~/runs/lora_s0/weights.pt --partition $p --wise-alpha $a --out ~/runs/lora_s0/wise_$a/$p.npz --workers 6; done
done
echo done > ~/runs/lora_s0/wise_DONE
