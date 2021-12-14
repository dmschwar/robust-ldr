This repository contains code to reproduce/examine the results of the paper entitled [Bolstering Adversarial Robustness with Latent Disparity Regularization](http://engr.arizona.edu/~dmschwar/papers/IJCNNPublishedVersion.pdf) ,  published in IJCNN 2021.

To begin with a pre-trained model, adjust the parameters in fine\_tuning\_experiment.py to your preferences (similar parameterizations to the paper's are already entered) and run the same file. 

To start from scratch, adjust the parameters in pre\_training.py and fine\_tuning\_experiment.py, then run the pre-training script. Modify the variable 'unprotectedModelName' to match the name of the undefended model generated by pre\_training.py, and run fine\_tuning\_experiment.py. 

Calculation of the regularization term introduced in this work is illustrated in the figure below, where rectangular solids graphically represent the arrays of activations of intermediate layers ( ![math](https://latex.codecogs.com/svg.latex?%5Cbold%7Bs%7D_k) ) of a convolutional neural network stimulated with a benign sample,  ![math](https://latex.codecogs.com/svg.latex?%5Cbold%7Bx%7D) , and its adversarial counterpart,  ![math](https://latex.codecogs.com/svg.latex?%5Cbold%7Bx%7D_%5Ctext%7Ba%7D) 
<!-- ![Why didn't my image load?](images/hldrIllustration.png) -->
<p align="center">
<img src="images/hldrIllustration.png" width="750"/>
</p>


Please contact me at schwartz.david.michael at gmail dot com for questions or links to pre-trained models.
