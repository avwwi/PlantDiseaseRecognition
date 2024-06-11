# PlantDiseaseRecognition
The PlantDiseaseRecognition project involves creating a model to classify images of plant leaves into various categories based on the type of disease or health status.

Dataset from Kaggle : https://www.kaggle.com/code/mohidabdulrehman/plant-disease-diagnosis

![image.png](https://drive.usercontent.google.com/download?id=1pgasjSLiB_2TKHJW4O1XSIenoK9R6sKp)

Lets train a convolutional neural network to work on the images of plant leaves and categories them as one of the below
1. Healthy
2. Powdery
3. Rust

### Workflow Overview
1. Data Preprocessing:
    * Resized images to a uniform size (i.e., 64X64).
    * Normalized pixel values.
    * Performed data augmentation (e.g., rotation, flipping, zooming) to increase the diversity of the training data and prevent overfitting.

2. Model Selection and Architecture:
    * Model Choice: Start with a simple CNN architecture and experiment with more complex ones (i.e. Tensorflow).
    * Architecture Design:
        * Input layer for image data.
        * Two convolutional layers with ReLU activation and max-pooling layers.
        * Fully connected (dense) layers for classification.
        * Output layer with 'softmax' activation for multi-class classification.

4. Training the Model:
    * Splitting Data: Dataset is already divided for training and testing, additional images for validation are also available.
    * Compiling the Model: Used an optimizer like 'Adam', and a loss function like 'categorical cross-entropy'.
    * Training Process:
        * Trained the model on the training set.
        * Validated on the validation set to tune hyperparameters and prevent overfitting.
        * Monitored training with metrics like accuracy and loss.
5. Evaluation and Optimization:
    * Testing: Evaluate the model's performance on the test set using accuracy, precision, recall, and F1-score.
    * Optimization:
        * Fine-tune hyperparameters (e.g., learning rate, batch size).
        * Experiment with deeper architectures or different activation functions.

## Code information

1. Data Preprocessing :
    * Image Augmentation : Using 'ImageDataGenerator' we accomplished adding simple geometric transformation in the training images, created batches of 32 Images and resized to 64 x 64px.
    * Feature scaling is also done by 'rescale = 1./255' so all the values are normalized.

2. Building the CNN:
    * Using tf.keras.Sequential() we have created a neural network with the below structure:
        * Convolution and Pooling layer : 'tf.keras.layers.Conv2D' is used with 32 filters, and feature detector size of 3x3 and activation function RelU as the convolution layer and 'tf.keras.layers.MaxPool2D' is used as the pooling layer ( size 2x2 ).
        * Second Convolution and Pooling layer.
        * Flattening : using 'tf.keras.layers.Flatten'.
        * Full Connection : using 'tf.keras.layers.Dense' we have added 128 hidden layers, activation function = 'RelU'.
        * Output Layer : using 'tf.keras.layers.Dense' with 3 units ( For 3 categorical dependant variable values), activation function = 'softmax'.

        ![image.png](https://drive.usercontent.google.com/download?id=1AApWGWRMLm3w_wPMEayzIIQuydJN0qTJ)


3. Compile and Build :
    * Optimizer = 'Adam'.
    * Loss function = 'Categorical cross entropy'.
    * Metrics = 'Accuracy'.
    * Epochs = '25'.

## Observation 
![image.png](https://drive.usercontent.google.com/download?id=1EF1KjVnOK0XHYzSrjEU4PJ2PTpkiykMm)

Over the epochs we can see the accuracy of the model rise up to 97% and the loss getting as low as 10%
| Epoch | Accuracy | Loss   |
|-------|----------|--------|
| 1     | 0.4609   | 1.1036 |
| 2     | 0.6906   | 0.6899 |
| 3     | 0.7626   | 0.5593 |
| 4     | 0.8095   | 0.482  |
| 5     | 0.8355   | 0.429  |
| 6     | 0.8434   | 0.3822 |
| 7     | 0.8802   | 0.3312 |
| 8     | 0.9043   | 0.271  |
| 9     | 0.911    | 0.2808 |
| 10    | 0.929    | 0.2039 |
| 11    | 0.9157   | 0.2317 |
| 12    | 0.9222   | 0.2246 |
| 13    | 0.9313   | 0.2228 |
| 14    | 0.9447   | 0.1643 |
| 15    | 0.9266   | 0.2007 |
| 16    | 0.9344   | 0.1988 |
| 17    | 0.9436   | 0.1455 |
| 18    | 0.9527   | 0.1401 |
| 19    | 0.9291   | 0.2108 |
| 20    | 0.9635   | 0.1133 |
| 21    | 0.9537   | 0.1162 |
| 22    | 0.9675   | 0.1055 |
| 23    | 0.9701   | 0.1006 |
| 24    | 0.9637   | 0.1129 |
| 25    | 0.9708   | 0.1043 |