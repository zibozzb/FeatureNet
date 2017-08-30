# FeatureNet
We developed a novel framework using Deep 3D Convolutional Neural Networks (3D-CNNs) termed FeatureNet to learn machining features from CAD models of mechanical parts. FeatureNet learns the distribution of complex machining feature shapes across a large 3D model data set and discovers distinguishing features that help in recognition process automatically. To train FeatureNet, a large-scale mechanical part datasets of 3D CAD models with labeled machining features is synthetically constructed. 

# Dataset
We create our own dataset by using **Solidworks** API which includes 24000 models belonging to 24 classes. And we uploaded our dataset in: [Dataset](https://github.com/zibozzb/Machining-feature-dataset)
![Database](https://github.com/zibozzb/FeatureNet/blob/master/img/2%20(1).png)

# Recognizer
We proposed a deep 3D convolutional neural network to be our recognizer. The input of recognizer is the model with only single feature. And the output is the class input feature belonging to.
![FeatureNet](https://github.com/zibozzb/FeatureNet/blob/master/img/1.png)

# Decomposition & Segmentation
We using **scikit-image** library to perform these two tasks. Decomposition will find different areas which are seperated from each other. Segmentation will seperate overlapping features.

# Result
* 1 : **Convergence of loss function and accuracy**
![convergence](https://github.com/zibozzb/FeatureNet/blob/master/img/Picture1.png)
* 2 : **Confusion matrix**
![confMatrix](https://github.com/zibozzb/FeatureNet/blob/master/img/2.png)
* 3 : **Three test cases**
![test](https://github.com/zibozzb/FeatureNet/blob/master/img/1%20(1).png)

*****************************************************************
### Owner:
	MAD Lab
	Department of Mechanical and Aerospace Engineering
	University at Buffalo, Buffalo, NY - 14260
	http://madlab.eng.buffalo.edu/
*****************************************************************
