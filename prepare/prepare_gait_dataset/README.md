
为了加速计算，应该先处理图片，然后保存成文件，然后再读取文件，这样的话，可以加快计算速度。
也就是说需要想一个文件处理过后应该是什么样子的。

例如,只使用患者步行的前半阶段来进行训练的话，
如果只把一个患者的不行周期分为两部分来看的话，

📓图像增强的操作，从pt文件中读取之后再进行也是可以的

```
whole dataset:
train:
ASD:
    one patient:
        1_gait.pt
        3_gait.pt
        5_gait.pt
    two patient:
        1_gait.pt
        3_gait.pt
        5_gait.pt
non ASD:
    one patient:
        1_gait.py
        2_gait.py
        3_gait.py
    two patient:
        1_gait.pt
        3_gait.pt
        5_gait.pt

val:
不区分患者的不行阶段
```

# Usage
程序的入口是main.py
