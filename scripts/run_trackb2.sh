#!/bin/bash
# Track B driver v2 (GPU exports). Seed 0 is already trained: export, WiSE sweep, then seeds 1-2, then CNN baselines.
cd ~/tw; export PYTHONPATH=src; M=~/data/manifests/dfire_train.csv; R=~/data/dfire
exp() { for p in val cal test xsrc_aof; do python3 -m fire_assessor.train export --manifest $M --root $R --weights ~/runs/lora_s$1/weights.pt --partition $p --out ~/runs/lora_s$1/$p.npz --workers 4 >> ~/runs/lora_s$1.log 2>&1; done; }
exp 0
for a in 0.0 0.25 0.5 0.75 1.0; do mkdir -p ~/runs/lora_s0/wise_$a
  for p in val test xsrc_aof; do python3 -m fire_assessor.train export --manifest $M --root $R --weights ~/runs/lora_s0/weights.pt --partition $p --wise-alpha $a --out ~/runs/lora_s0/wise_$a/$p.npz --workers 4 >> ~/runs/wise.log 2>&1; done; done
echo done > ~/runs/lora_s0/wise_DONE
for s in 1 2; do python3 -m fire_assessor.train train --manifest $M --root $R --out ~/runs/lora_s$s --epochs 10 --batch-size 32 --seed $s --workers 4 > ~/runs/lora_s$s.log 2>&1; exp $s; done
for arch in resnet50 efficientnet_b0; do for s in 0 1 2; do python3 scripts/train_cnn.py --arch $arch --manifest $M --root $R --out ~/runs/${arch}_s$s --seed $s > ~/runs/${arch}_s$s.log 2>&1; done; done
echo ALLDONE > ~/runs/trackb.done
