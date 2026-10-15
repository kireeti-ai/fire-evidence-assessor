#!/bin/bash
# CNN baselines after generation has released the GPU (the first attempt OOM'd while Kolors held ~20 GB).
cd ~/tw; export PYTHONPATH=src; M=~/data/manifests/dfire_train.csv; R=~/data/dfire
until [ -f ~/synth/GEN_DONE ]; do sleep 30; done
for arch in resnet50 efficientnet_b0; do for s in 0 1 2; do
  python3 scripts/train_cnn.py --arch $arch --manifest $M --root $R --out ~/runs/${arch}_s$s --seed $s > ~/runs/${arch}_s$s.log 2>&1
done; done
echo done > ~/runs/cnn.done
