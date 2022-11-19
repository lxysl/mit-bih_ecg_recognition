# A Deep Learning Baseline of MIT-BIH ECG Recognition

* Details of this project can be found in my blog articles:

[使用Python+TensorFlow2构建基于卷积神经网络（CNN）的ECG心电信号识别分类（一）](https://www.cnblogs.com/lxy764139720/p/12830037.html)

[使用Python+TensorFlow2构建基于卷积神经网络（CNN）的ECG心电信号识别分类（二）](https://www.cnblogs.com/lxy764139720/p/12831422.html)

[使用Python+TensorFlow2构建基于卷积神经网络（CNN）的ECG心电信号识别分类（三）](https://www.cnblogs.com/lxy764139720/p/12840183.html)

[使用Python+TensorFlow2构建基于卷积神经网络（CNN）的ECG心电信号识别分类（四）](https://www.cnblogs.com/lxy764139720/p/12879907.html)

* Requirements:

common packages:

```shell
pip install wfdb, pywavelets, numpy, matplotlib, seaborn, sklearn, tensorboard
```

tensorflow packages:

```
tensorflow>=2.0.0
```

pytorch packages:

```
torch>=1.4.0
```

* Usage:

for tensorflow:

```shell
python main_tf.py
```

for pytorch:

```shell
python main_torch.py
```

visualize the log in tensorboard:

```shell
tensorboard --logdir=./logs/<log_dir_name>
```

* File Structure:

main_tf.py: code implemented by TensorFlow2

main_torch.py: code implemented by PyTorch

utils.py: util functions about data processing and plotting

* Note:

**Be careful!** The input of the convolution (1d) layer in tensorflow and pytorch are different.

In tensorflow it is [batch_size, length, channel], while in pytorch is [batch_size, channel, length].

The denoised ECG data shape in numpy format is [batch_size, length].

Therefore, the input of the convolution layer in tensorflow needs to be reshaped to [batch_size, length, channel], while that in pytorch needs to be reshaped to [batch_size, channel, length], where channel equals 1.

Data shape in this version of code may vary with my blogs.

* Feel Helpful?

If you find this project helpful, please give me a star. Thanks!

If you could tip me a cup of coke, I would be very grateful!

![IMG_3509.JPG](https://s2.loli.net/2022/11/11/shdDtlFcv8oaWPO.jpg)
