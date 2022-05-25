# Bootstrapped-MAE
Improve MAE by bootstrapping
何恺明博士在Masked autoencoders are scalable vision learners【1】一文中架起了CV与NLP之间的桥梁，指明了通往CV大模型的方向。其中提出的
MAE预训练模型在分类等下游任务上取得了很好的效果。但上述模型存在着改进的方向，比如可以将预训练模型的重建目标修改为MAE的encoder输出
的高级特征，相比于原论文重建像素来讲，由于encoder输出的是更加高级的语义信息，所以可能会产生更好的训练效果（此想法来源于清华大学高阳老师）
本项目对上述思路进行了代码实现，具体如下：

Bootstrapped MAE.py是用来实现boostrappedMAE算法的主文件，engine_pretrain_bootstrappedMAE.py
models_reconstruct.py、models_mae.py用来定义实现该算法所需的子模块。

finetune_all.py和linpro_all.py是我编写的用来测试所有预训练模型效果的文件，其可以自动遍历对应文件夹下的预训练模型进行测试。

util文件夹相较于MAE官方代码没有太大改动，其用来定义学习率，位置向量，数据预处理等操作。
唯一的区别在于我自定义了一个misc_bootstrappMAE.py用来实现对于bootstrapMAE模型的保存与提取。

main_finetune.py和main_linprobe.py是用来对单个预训练模型进行测试的文件
checkpoint存储着我根据bootstrappedMAE算法预训练的模型——checkpoinit-bootMAE。checkpoint-MAE为原始MAE算法的预训练模型。
修改main_finetune.py和main_linprobe.py的模型路径参数可以对
该预训练模型的效果进行测试。

main_pretrain.py是原始的MAE算法的预训练文件。

本次的代码实现以MAE的官方代码为基础改编而来，所有包含中文注释的地方以及新增的文件即为本人的改动之处。
[1] He, Kaiming, Xinlei Chen, Saining Xie, Yanghao Li, Piotr Dollár, and Ross Girshick. "Masked autoencoders are scalable vision learners." arXiv preprint arXiv:2111.06377 (2021).
