# Image Classification

the simplest way to setup an image classification app.



## Prepare data

Download data.

## Train

```
python train.py
```

You will see:

```
Found 2000 images belonging to 2 classes.
Found 1000 images belonging to 2 classes.
2020-06-28 15:41:47.490966: I tensorflow/stream_executor/platform/default/dso_loader.cc:44] Successfully opened dynamic library libcuda.so.1
2020-06-28 15:41:47.523359: I tensorflow/stream_executor/cuda/cuda_gpu_executor.cc:981] successful NUMA node read from SysFS had negative value (-1), but there must be at least one NUMA node, so returning NUMA node zero
2020-06-28 15:41:47.523655: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1561] Found device 0 with properties: 
pciBusID: 0000:01:00.0 name: GeForce GTX 1080 Ti computeCapability: 6.1
coreClock: 1.582GHz coreCount: 28 deviceMemorySize: 10.91GiB deviceMemoryBandwidth: 451.17GiB/s
2020-06-28 15:41:47.524008: W tensorflow/stream_executor/platform/default/dso_loader.cc:55] Could not load dynamic library 'libcudart.so.10.1'; dlerror: libcudart.so.10.1: cannot open shared object file: No such file or directory; LD_LIBRARY_PATH: /usr/local/lib:/usr/local/cuda/lib64:/media/fagangjin/samsung/source/ai/inno/maskrcnn_onnx_pipe/onnx-tensorrt/build:/media/fagangjin/samsung/source/ai/inno/maskrcnn_onnx_pipe/maskrcnn_trt_cc/build:/home/fagangjin/TensorRT/lib:/home/fagangjin/TensorRT/targets/x86_64-linux-gnu/lib:/home/fagangjin/work/trt_engine_check/build:
2020-06-28 15:41:47.525026: I tensorflow/stream_executor/platform/default/dso_loader.cc:44] Successfully opened dynamic library libcublas.so.10
2020-06-28 15:41:47.526037: I tensorflow/stream_executor/platform/default/dso_loader.cc:44] Successfully opened dynamic library libcufft.so.10
2020-06-28 15:41:47.526193: I tensorflow/stream_executor/platform/default/dso_loader.cc:44] Successfully opened dynamic library libcurand.so.10
2020-06-28 15:41:47.527344: I tensorflow/stream_executor/platform/default/dso_loader.cc:44] Successfully opened dynamic library libcusolver.so.10
2020-06-28 15:41:47.527986: I tensorflow/stream_executor/platform/default/dso_loader.cc:44] Successfully opened dynamic library libcusparse.so.10
2020-06-28 15:41:47.530480: I tensorflow/stream_executor/platform/default/dso_loader.cc:44] Successfully opened dynamic library libcudnn.so.7
2020-06-28 15:41:47.530501: W tensorflow/core/common_runtime/gpu/gpu_device.cc:1598] Cannot dlopen some GPU libraries. Please make sure the missing libraries mentioned above are installed properly if you would like to use GPU. Follow the guide at https://www.tensorflow.org/install/gpu for how to download and setup the required libraries for your platform.
Skipping registering GPU devices...
2020-06-28 15:41:47.530688: I tensorflow/core/platform/cpu_feature_guard.cc:143] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 FMA
2020-06-28 15:41:47.551883: I tensorflow/core/platform/profile_utils/cpu_utils.cc:102] CPU Frequency: 3696000000 Hz
2020-06-28 15:41:47.552429: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x563e22810d40 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2020-06-28 15:41:47.552443: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
2020-06-28 15:41:47.553606: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1102] Device interconnect StreamExecutor with strength 1 edge matrix:
2020-06-28 15:41:47.553616: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1108]      
WARNING:tensorflow:From train.py:73: Model.fit_generator (from tensorflow.python.keras.engine.training) is deprecated and will be removed in a future version.
Instructions for updating:
Please use Model.fit, which supports generators.
Epoch 1/15
15/15 [==============================] - ETA: 0s - loss: 0.8145 - accuracy: 0.4963^CTraceback (most recent call last):
```

