# anime_audio_classification_model
This model aims to separate one character's audio files that you need from other characters. That is, sort out the voice files of character 1 into a specific folder you like, and put other characters' voice files into another folder.  
Extract features by MFCC, and then choose which model you would like to use among SVM, ANN, and CNN.

This readme is expected to be completed after the codes are totally done.

这个模型的目的是，在一集动画所有的角色台词中，分类出属于某个特定角色的台词。
用法步骤如下：
1.
2.
3.

实现原理
使用torch.transform.MFCC提取特征，然后选择其中一个模型训练：SVM，ANN，CNN。
