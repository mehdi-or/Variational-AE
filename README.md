# Dimentionality of feature space between pre-frontal cortex and Ventral temporal cortex using a hybrid netweork of variational auto encoders and a logistic classifier
## 1. Introduction
Sensory information coming form outside world is encoded by networks of neurons in the brain. Neural decoding is an emerging field in neuroscience that tries to decode the information that is already encoded by the brain. This helps to better understand how the brain processes sensory information in order to evaluate the environment and make decisions accordingly. This method has different applications. For instance, Taschereau-Dumouchel et.al used this technique to unconsciously rewire the brain of patients who have phobia associated with a specific animal/object in order to treat their anxiety disorder[1] (eg. snake, spider). To build this decoder, typically, the feared object is shown to the subject while the fMRI images are being acquired. One drawback of this techniques is that patients are feared from those objects and as a result of that they may withdraw from the treatment plan. In order to address this issue, researchers used a newly developed techniques called hyperalignment [2, 3]. This way, it is possible to obtain the relevant representation of feared objects by showing them to the surrogate subjects.
In this work we are interested in finding the features that are important to specific regions of brain (eg. VT and PFC) and how these brain regions share information  and what is the dimension of feature space between theses regions.


## 2. Method
### pre-processing of the data-sets
Functional Magnetic Resonance Imaging (fMRI) technique was used to collect brain scans of 80 subjects while showing them the pictures of 40 different categories including 30 categories of animals (mammals, reptiles, insects, birds, etc), and 10 categories of man-maid objects (hammer, chair, etc). Each category (class) contains 90 different images (eg. dog1 is a different picture from dog2) adding up the total number of pictures to 3600. These picture were shown to each subject in chunks of 2, 3, 4, or, 6 images of the same category. Each chunk of a specific category is called a mini-block (Figure 1) . These data were collected in 6 runs with short breaks for each subject by showing them 600 images in each run. In order to increase the attention to each category, subjects were asked to press a button once a category changed. The order of image presentation was pseudorandomized but fixed across all subjects. Each picture was shown on the screen for 0.98 s and the repetition time (TR) of the scanner was 2 s. Therefore, in each scan two pictures were shown to subjects. This potentially can lead to contamination of the data in such a way that in one TR two images from two different categories were acquired.

![](/images/1.png)

figure 1- Taschereau-Dumouchel et.al PNAS 2018

Multivoxel patterns from pre-frontal cortex (PFC) and ventral temporal cortex (VT) of each subject were selected. In order to investigate whether the data were contaminated between category change, a simple neural network architecture with one hidden layer that was used to build the decoder (figure 2). The number the features (voxels) in the impute layer was different for each person (due to the natural variation of the size of different areas of brain in different people), the number of the hidden units in hidden layer was 500 features and the output layer was defined by 40 different categories. The activation function for each layer was defined as the sigmoid function.
One-Versus-all classification technique was used to decode voxels for each of 40 classes. To test the actual accuracy of the decoder, a five-fold cross-validation was used for each (training over 80% of the data and testing on 20% of the data for 5 times). The accuracy was considered as the mean of the ratio of the correct guesses to the total test exemplars.
![](/images/3.png)
Figure 2.

Since Neural Network is a mathematical algorithm which tries to minimize the error for the cost function, it is important to feed the feature to the algorithm in such a way that the corresponding outputs do not follow a specific pattern. To break the repetitive pattern in the mini-blocks, the exemplars were randomly shuffled across the entire 3600 exemplars for each subject. It was observed that most of incorrect classification occur within a category change (the last exemplar of the former mini-black and the first exemplar of the later mini-block). This is probably due to the fact that in each TR, 2 images were shown to subjects and during the transition from a one mini-block to the other, the subject sees two pictures from two different categories. As a result, the scan during the transition from one class to the other will contain some voxels that are related to former class and some that are related to later class. In order to test this hypothesis, we reconstructed 5 different modified data-sets as follow:
##### 1. Removing only the last exemplar of each mini-block
##### 2. Removing only the fist exemplar of each mini-block
##### 3. Removing only the last and the first exemplar of each mini-block
##### 4. Removing one exemplar within each mini-block that is not the first of the last one (except those mini-block that has only two exemplars)
##### 5. Not removing any data
Then we are able to find weather the accuracy for each of these newly modified data-sets (each group) is significantly different. In other words, is there a main effect on the accuracy due to the presence/absence of first and last exemplar of each mini-block?
The other factor that can potentially influence the accuracy of the decoder is the region of interest (ROI) in the brain (eg. PFC and VT).
As depicted in figure 3, even though by removing the contaminated trials we lose almost 25% of the data, we get a higher accuracy than not removing any data. Also, it is evident that removing a good trial will decrease the classification accuracy.	
![](/images/contamination.jpg)
Figure 3.

Due to the contamination in the data, the last trial of the former mini-block, just before the change in the category happens, was removed. In addition to that we had some none informative features in the voxel activities of VT and PFC (features with zero values that are considered non informative) that were removed from data-sets.

### Network architecture
In order to establish a the information flow between VT and PFC two variational auto-encoders (VAEs) were used to predict the voxel activities of PFC based on VT and vice versa. In order to force the network to encode the most relevant features in the bottleneck of the network, a simple classifier using a logistic regretion was used to classify each trial to one of the 40 classes. The arcitechture of the network is show in figure 4. The cost function for the VAEs were defined as Mean-Square-Error (MSE) and the cost function for the classifier was defines as Categorical Cross-Entropy (CCE). Then these cost function were trained simultaneously. After trying different number of features for the bottleneck of the network a feature number of 20 was considered for the bottleneck. 
The activation function for the hidden layers was Tanh, for output layer of the VAEs was a linear function and for the output of the classifier was the soft-max function. In order to prevent of over-fitting a drop-out of 20% was defined after each hidden  layer. 

![](/images/architecture.png)
figure4.

70 percent of the data-set was using to train the networks, 20 percent of the data-set was for cross-validation and 10 percent of the data-set was used as test-set. Adam optimizer was used to minimize the cost functions and to converge faster, features from VT and PFC were scaled between -1 and 1.
Following of the training, using the test-set, the activity of VT, PFC and features in the bottleneck were predicted.

## 3.Result

The Pearson correlation of predicted values of PFC and VT for the same trials  were obtained. Figure 5 shows an example of the correlation between predicted PFC and the actual PFC (here a Pearson correlation of 0.66). 

![](/images/Figure_3.png)
figure5.

A cononical correlation analysis (CCA) with 20 components (the same number of feature in the bottleneck) was used to in order to evaluate the performance of this netwrok. As it can be seen in figure 6, the correlation of of the predicted PFC of the same trial is much lower (about 0.29). The mean value of the correlation between the actual vs predicted values, using CCA, was approximately 0.25 which was significantly lower than our noble network architecture which was approximately 0.65. These result shows that the performance of the VAEs-classifier network is better than CCA.

![](/images/Figure_7.png)
figure 6.

Despite of using a linear activation function in the output of VAEs, it seems that the network cannot follow the extreme values of VT and PFC. As it is depicted it figure 7, it appears that the network is treating the values that are more than 2 standard-deviation above the mean as noise.
 
![](/images/Figure_2.png)
figure 7.

Figure 8. shows the predicted and actual values of the same trial. It can be seen that prediction is following the same trend as the actual values. This plot also shows that the network has difficulties in predicting the values that are far from the mean.

![](/images/Figure_1_2.png)
figure 8.

In order to see if there has been any correlation between that voxel activities of VT and PFC between different trials of the same class, the average of the Pearson for all trial was calculated for VT and PFC. Figure 9 shows that there is almost no correlation between different trials of the same class. This is probably due to the fact the dimensionality of the VT and PFC are very high and the signal to noise (SNR) is low.

![](/images/Figure_4.png)
figure 9.

By looking at the features in the bottleneck only, it is expected that the network is picking the most relevant features and that may reveal some similarities between different trails of the same class. Figure 10. depicts that the average correlation within the same class is between 2 to 3 times more than the average correlation between the different classes.

![](/images/Figure_6.png)
figure 10.
