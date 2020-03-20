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
