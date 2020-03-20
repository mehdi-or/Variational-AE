# Dimentionality of feature space between pre-frontal cortex and Ventral temporal cortex using a hybrid netweork of variational auto encoders and a logistic classifier
## Introduction
The sensory information coming form outside world is encoded by networks of neurons in the brain. Neural decoding is an emerging field in neural science that tries to decode the information that is already encoded by the brain. This helps to better understand how the brain processes the sensory information to make sense form the outside world. This method has different applications. For instance, Taschereau-Dumouchel et.al used this technique to unconsciously rewire the brain of patients who have phobia associated with a specific animal/object in order to treat their anxiety disorder[1] (eg. snake, spider). To build this decoder, typically, the feared object is shown to the subject while the fMRI images are being acquired. One drawback of this techniques is that patients are feared from those objects and as a result of that they may withdraw from the treatment plan. In order to address this issue, researchers used a newly developed techniques called hyperalignment [2, 3]. This way, it is possible to obtain the relevant representation of feared objects by showing them to the surrogate subjects. The other application can be using the information from different regions of the brain to implement decoding algorithms in order to find the categorical object representation of different regions of the brain.

![](/images/1.png)
figure 1- Taschereau-Dumouchel et.al PNAS 2018

## Method
Functional Magnetic Resonance Imaging (fMRI) technique was used to collect brain scans of 70 subjects while showing them the pictures of 40 different categories including 30 categories of animals (mammals, reptiles, insects, birds, etc), and 10 categories of man-maid objects (hammer, chair, etc). Each category (class) contains 90 different images (eg. dog1 is a different picture from dog2) adding up the total number of images to 3600 pictures that were shown to each subject in chunks of 2, 3, 4, or, 6 images of the same category. Each chunk of a specific category is called a mini-block. These data were collected in 6 runs with short breaks for each subject by showing them 600 images in each run. In order to increase the attention to each category, subjects were asked to press a button once a category changed. The order of image presentation was pseudorandomized but fixed across all subjects. Each picture was shown on the screen for 0.98 s and the repetition time (TR) of the scanner was 2 s. Therefore, in each scan two pictures were shown to subjects. This potentially can lead to contamination of the data in such a way the in one TR two images from two different categories were acquired.
Multivoxel patterns from pre-frontal cortex (PFC) and ventral temporal cortex (VT) of each subject were selected. A simple neural network architecture with one hidden layer was used to build the decoder. The number the features (voxels) in the impute layer was different for each person (due to the natural variation of the size of different areas of brain in different people), the number of the hidden units in hidden layer was 500 features and the output layer was defined by 40 different categories. The activation function for each layer was defined as the sigmoid function.
In order to prevent the overfitting of the data a regularization constant of  was chosen. One-Versus-all classification technique was used to decode voxels for each of 40 classes. To test the actual accuracy of the decoder, a five-fold cross-validation was used for each (training over 80% of the data and testing on 20% of the data for 5 times). The accuracy was considered as the mean of the ratio of the correct guesses to the total test exemplars.

Where CG is the number of correct guesses and T is the total number of test exemplars.
Since Neural Network is a mathematical algorithm which tries to minimize the error for the cost function, it is important to feed the feature to the algorithm in such a way that the corresponding outputs do not follow a specific patter. To break the reparative pattern in the mini-blocks, the exemplars were randomly shuffled across the entire 3600 exemplars for each subject. It was observed that most of wrong guesses occurs within a category change (the last exemplar of the former mini-black and the first exemplar of the later mini-block). This is probably due to the fact that in each TR, 2 images were shown to subjects and during the transition from a one mini-block to the other, the subject sees two pictures from two different categories. As a result, the scan during the transition from one class to the other will contain some voxels that are related to former class and some that are related to later class. In order to test this hypothesis, we reconstructed 5 different modified datasets as follow:
    1. Removing only the last exemplar of each mini-block
    2. Removing only the fist exemplar of each mini-block
    3. Removing only the last and the first exemplar of each mini-block
    4. Removing one exemplar within each mini-block that is not the first of the last one (except those mini-block that has only two exemplars)
    5. Not removing any data
Then we are able to find weather the accuracy for each of these newly reconstructed datasets (each group) is significantly different. In other words, is there a main effect on the accuracy due to the presence/absence of first and last exemplar of each mini-block?
The other factor that can potentially influence the accuracy of the decoder is the region of the brain that we select the data from. To test this hypothesis, we looked at the PFC and VT cortex to see if there is a main effect due to the region of interest (ROI). 
To test for theses hypotheses, a two-way repeated measures ANOVA can be used. The reason for choosing this test is that we have two factors within subjects, one is the ROI which has two factor levels (PFC and VT) and the other factor is data removal (which has 5 factor levels). Then, we look at the accuracy of the decoder for all of the combinations of these factor levels in 10 of the participants so that we are able to see whether there is a main effect due to each factor. Also, we can test whether there is an interaction between the two factors.
