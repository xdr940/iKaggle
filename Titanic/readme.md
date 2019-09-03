用多层感知机做的，得分0.76555,只留下pclass、sex、Age、fare四个特征做预测，其中年龄空白就补0
然后四个特征的随机变量都归一化处理

![](https://github.com/xdr940/iKaggle/blob/master/Titanic/pics/MLP_result.png)


更新

之前用的是01normalization ，这次改成z-score标准化。。。。。更低了0.7165

![](https://github.com/xdr940/iKaggle/blob/master/Titanic/pics/pic2.png)

关于标准化的策略等超参数还是靠经验吧,先不问了。

![](https://github.com/xdr940/iKaggle/blob/master/Titanic/pics/MLP_result2.png)




# 文件

main.py 主文件
net.py 是网络结构
PreProcessing.py 数据预处理

train.csv 官网给的训练集文件
test.csv 官网给的需要做预测的数据集
out.csv 工程生成的待提交文件
historyfile.bin 网络训练过程, 用来做acc 和 loss 函数的图
modelweight.model 网络参数文件