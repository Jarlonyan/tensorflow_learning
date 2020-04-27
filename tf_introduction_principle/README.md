
# 《深度学习之TensorFlow入门、原理与进阶实战》

+ requirements:
    + tensorflow 1.4.0
    + matplotlib 1.3.1

+ 参考：
    + 代码：https://github.com/Lebhoryi/learning_tf
    + 数据：https://github.com/InsaneLife/CIFAR_TensorFlow.git

    ```
    import cifar10
    cifar10.maybe_download_and_extract()
    ```

+ 解决错误
    + `module' object has no attribute 'per_image_whitening'` : 
    `per_image_whitening` 改为：`per_image_standardization`


