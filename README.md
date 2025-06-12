# DCGAN_MNIST
复现 paper《UNSUPERVISED REPRESENTATION LEARNING WITH DEEP CONVOLUTIONAL GENERATIVE ADVERSARIAL NETWORKS》

# DCGAN.py：
DCGAN网络结构，main文件

# ReadInfo.py：
将保存的txt内容以图像方式显示，例如损失函数曲线，学习率曲线

# D/xxx.pth：
鉴别器权重，使用model.load_state_dict(torch.load(PATH))

# G/xxx.pth：
生成器权重，使用model.load_state_dict(torch.load(PATH))

# fake_imgs/xxx.jpg：
每个epoch生成的图像

# INFO.txt
训练过程中保存的训练信息，使用ReadInfo.py读取，需要把文件路径修改掉
