
+ requirements:
    + tensorflow 1.4.0
    + matplotlib 1.3.1

+ 参考：
    + 代码：https://github.com/Lebhoryi/learning_tf
    + 数据：https://github.com/InsaneLife/CIFAR_TensorFlow.git

+ 解决错误
    + `module' object has no attribute 'per_image_whitening'`
    `per_image_whitening` 改为：`per_image_standardization`


