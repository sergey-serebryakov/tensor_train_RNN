# P100 PCIe
```bash
docker build -f ./docker/Dockerfile-gpu -t deepts/tensorflow:gpu-1.13-1 .
nvidia-docker run --rm -ti -v $(pwd):/workspace deepts/tensorflow:gpu-1.13-1
cd /workspace
CUDA_VISIBLE_DEVICES=0 python ./test_trnn.py  --training_steps 10000 --display_step 500
```

```bash
Step     1, iter time = 73.3234 seconds, throughput = 340.96 seq/s, minibatch loss = 0.3538, validation loss = 0.3538
Step   500, iter time = 90.2906 seconds, throughput = 276.88 seq/s, minibatch loss = 0.1651, validation loss = 0.1512
Step  1000, iter time = 88.4070 seconds, throughput = 282.78 seq/s, minibatch loss = 0.1477, validation loss = 0.1186
Step  1500, iter time = 87.7228 seconds, throughput = 284.99 seq/s, minibatch loss = 0.1063, validation loss = 0.0969
Step  2000, iter time = 88.6647 seconds, throughput = 281.96 seq/s, minibatch loss = 0.0800, validation loss = 0.0729
Step  2500, iter time = 88.5542 seconds, throughput = 282.31 seq/s, minibatch loss = 0.0684, validation loss = 0.0727
Step  3000, iter time = 88.3127 seconds, throughput = 283.08 seq/s, minibatch loss = 0.0583, validation loss = 0.0624
Step  3500, iter time = 88.0270 seconds, throughput = 284.00 seq/s, minibatch loss = 0.0565, validation loss = 0.0563
Step  4000, iter time = 88.2877 seconds, throughput = 283.17 seq/s, minibatch loss = 0.0523, validation loss = 0.0521
Step  4500, iter time = 88.1773 seconds, throughput = 283.52 seq/s, minibatch loss = 0.0367, validation loss = 0.0360
Step  5000, iter time = 88.6703 seconds, throughput = 281.94 seq/s, minibatch loss = 0.0351, validation loss = 0.0350
Step  5500, iter time = 88.1438 seconds, throughput = 283.63 seq/s, minibatch loss = 0.0290, validation loss = 0.0271
Step  6000, iter time = 88.2309 seconds, throughput = 283.35 seq/s, minibatch loss = 0.0426, validation loss = 0.0423
Step  6500, iter time = 88.3810 seconds, throughput = 282.87 seq/s, minibatch loss = 0.0242, validation loss = 0.0240
Step  7000, iter time = 88.3130 seconds, throughput = 283.08 seq/s, minibatch loss = 0.0153, validation loss = 0.0153
Step  7500, iter time = 88.3626 seconds, throughput = 282.92 seq/s, minibatch loss = 0.0143, validation loss = 0.0148
Step  8000, iter time = 88.8670 seconds, throughput = 281.32 seq/s, minibatch loss = 0.0101, validation loss = 0.0099
Step  8500, iter time = 87.9432 seconds, throughput = 284.27 seq/s, minibatch loss = 0.0141, validation loss = 0.0140
Step  9000, iter time = 88.1007 seconds, throughput = 283.77 seq/s, minibatch loss = 0.0114, validation loss = 0.0117
Step  9500, iter time = 87.9904 seconds, throughput = 284.12 seq/s, minibatch loss = 0.0105, validation loss = 0.0104
Step 10000, iter time = 88.1126 seconds, throughput = 283.73 seq/s, minibatch loss = 0.0100, validation loss = 0.0104
[INFO]     Optimization finished in 1840.8868 s, iter time 88.3780 s, throughput 282.88 sequences/s
[INFO]     batch size=50, input length=12, output length=88
[WARNING]  Iteration time includes time to run model validation.
Testing Loss: 0.010106621

```
