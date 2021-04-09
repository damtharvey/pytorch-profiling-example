# Requirements

You need PyTorch >= 1.8.1. Follow the instructions at https://pytorch.org/get-started/ to install it.

If you want to profile TensorRT, install torch2trt by following instructions at https://github.com/NVIDIA-AI-IOT/torch2trt. Otherwise, remove `import torch2trt` from `main.py`.

To view the jsons in TensorBoard, you need to 

```shell
pip install torch_tb_profiler
```

# Example
Profile:
```shell
python main.py --architecture resnet18 --results-directory example
```
View results:
```shell
tensorboard --logdir example
```