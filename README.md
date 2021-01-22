# AlexNet论文思维导图与代码实现
ImageNet Classification with Deep ConvolutionalNeural Networks（AlexNet）

mindmap.pdf为思维导图

# Python代码
## 1°AlexNet（MINST数据集）.py
该文件实现AlexNet对MINST的测试，在MINST测试集上准确率已达到99.01%
## 2°AlexNet(CIFAR-10).py
该文件实现AlexNet对CIFAR-10数据集的测试，在CIFAR-10训练集上的准确率已达到99.38%，在测试集上的准确率为86.67%，即存在过拟合现象，尚未解决，开放给coder们
## 3°AlexNet.py
该文件中有两个类：AlexNet与AlexNet1，前者是完全复现论文，后者取消了前者中的参数初始化。


在Alex那篇论文中，作者将全连接层偏置和第二、四、五个卷积层的偏置初始化为1


经过对MINST和CIFAR-10数据集的多次试验，可以发现：


如果像论文那样初始化参数，将会导致局部收敛，这种局部收敛还是在一个很低的水平上。


故在AlexNet1中取消了偏置的初始化，并在两个数据集上进行了测试

# Alex-MNIST.pkl和Alex-CIFAR-10.pkl 保存了已经训练好的模型的参数
