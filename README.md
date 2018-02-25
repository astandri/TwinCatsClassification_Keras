# Image Classification using keras

## Meet my twin cats, Claudia and Lucy

------------------------------------------------------------------------------

Claudia             |  Lucy
-------------------------|-------------------------
![Claudia](https://raw.githubusercontent.com/astandri/TwinCatsClassification_Keras/master/Images/Claudia/Claudia15.jpg)  |  ![Lucy](https://raw.githubusercontent.com/astandri/TwinCatsClassification_Keras/master/Images/Lucy/Lucy7.jpg)


## About Claudia:
She's so calm and kind. she always stand in front of my door, waiting for me to come home. Once she hear the sound of my motorcycle, she'll shout 'MIAWW MIAWW MIAAWWW!!!' and run her feet to me. She's beautiful


## About Lucy:
This one have a lot of energy. She can sing all night long! unlike her sister, she always angry when other cats touch her (even males). Well I guess that's why she's still single while her sister already had a kitten :'D


## About this project
I've just learned about Keras and I think I need to get started to have fun with this one. So, this is my 1st project using keras, in this project, I'll try to make a model to classify which one is Claudia, which one is Lucy.


![Keras](https://s3.amazonaws.com/keras.io/img/keras-logo-2018-large-1200.png)
##### image source: https://keras.io/

Well, let's begin!


### 1. **Data collection**:
I took 100 samples of photos for both of them manually, while sleeping, eating, walking, and even cuddling in my foot.


### 2. **Data Preprocessing**:
Using OpenCV with the helps from matplotlib and pandas, I preprocess the raw images to become a clean dataset stored in dataframe. 
I labeled the dataset with **Claudia** as **0** and **Lucy** as **1**. I then further split this dataset to 80:10:10 ratio for Testing purpose

The dataset then saved into three parts:

	1. train.csv (160 rows of observations)
	2. validation.csv (20 rows of observations)
	3. test.csv (20 rows of observations)


### 3. **Model Building**:
I created a class to automatically build a model with desired number of layers and nodes per layer using keras.
Defaulted to 1 layer and 10 nodes as base model.


### 4. **Training and Experimentation**:
Using my newly created class before, I did some experimentation in hyperparameters to build the best model. I use combination of [2,3,4] layers alongside with [100,150,200,250,300] nodes.

Here's the result:
![2 Layers](https://raw.githubusercontent.com/astandri/TwinCatsClassification_Keras/master/Images/experiment_with_2layers.PNG)
![3 Layers](https://raw.githubusercontent.com/astandri/TwinCatsClassification_Keras/master/Images/experiment_with_3layers.PNG)
![4 Layers](https://raw.githubusercontent.com/astandri/TwinCatsClassification_Keras/master/Images/experiment_with_4layers.PNG)

After the experimentation, the best 3 models is selected. they are:

Model             |  Training Loss | Validation Loss
-------------------------|-------------------------: | -------------------------:
2 Layers 150 Nodes | 0.2732 | 0.8059
3 Layers 150 Nodes | 0.1646 | 0.8059
4 Layers 100 Nodes | 0.2067 | 0.8059


### 5. **Testing**
Using 3 best models chosen, I do a prediction using test dataset. Here's the test result:

#### 5.1 Accuracy:
------------------------------------------------------------------------------

Model             |  Accuracy 
-------------------------|-------------------------: 
2 Layers 150 Nodes | 80%
3 Layers 150 Nodes | 80% 
4 Layers 100 Nodes | 75%


#### 5.2 ROC Curve
![Test Result](https://raw.githubusercontent.com/astandri/TwinCatsClassification_Keras/master/Images/test_result.PNG)
well, it seems the model with 2 layers and 150 wins the competition.

#### 5.3 Do some checks with model 2 layers and 150 nodes
------------------------------------------------------------------------------

Test index 0     |  Test Data Index 9 | Test Data Index 2
:-------------------------:| :-------------------------: | :-------------------------:
![index0](https://raw.githubusercontent.com/astandri/TwinCatsClassification_Keras/master/Images/Claudia/Claudia70.jpg) | ![index0](https://raw.githubusercontent.com/astandri/TwinCatsClassification_Keras/master/Images/Lucy/Lucy58.jpg) | ![index0](https://raw.githubusercontent.com/astandri/TwinCatsClassification_Keras/master/Images/Claudia/Claudia20.jpg)
Actual: Claudia (0) | Actual: Lucy (1) | Actual: Claudia (0)
Predicted: Claudia (0) | Predicted: Lucy (1) | Predicted: Lucy (1)
The machine's right!!! | The machine's right!!! | The machine's got it wrong now, maybe because Claudia too close to the camera? :'D

### 6. Conclusions
- This is obviously not a perfect model, Deep Learning with Neural Network usually works well with big datasets. using only 200 observations (160 as training data) to get 80% accuracy was good enough.
- This is my 1st project using keras, so I need to learn more utilizing it in the future and welcoming any advice

## So question for you, which one's cuter? Claudia? Lucy?
