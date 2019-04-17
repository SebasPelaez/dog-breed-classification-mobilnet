# DOG BREED CLASSIFICATION WITH MOBILNETS ARCHITECTURE

In this project I implemented a deep learning architecture with the objective to predict the dog's breed from a set of images.

For doing this project I used the following resources:

* Howard, A. G., Zhu, M., Chen, B., Kalenichenko, D., Wang, W., Weyand, T., ... & Adam, H. (2017). Mobilenets: Efficient convolutional neural networks for mobile vision applications. arXiv preprint arXiv:1704.04861.

* Chi-Feng Wang. (Aug 13, 2018). A Basic Introduction to Separable Convolutions. Towards Data Science. Recovered From: https://towardsdatascience.com/a-basic-introduction-to-separable-convolutions-b99ec3102728

* Zehaos. (Nov 5, 2017). MobilNet. Github. https://github.com/Zehaos/MobileNet

* Aditya Khosla, Nityananda Jayadevaprakash, Bangpeng Yao and Li Fei-Fei. Novel dataset for Fine-Grained Image Categorization. First Workshop on Fine-Grained Visual Categorization (FGVC), IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2011

## Project Structure

```
- tests/
    data.ipynb
    model.ipynb
    preprocessing.ipynb
- config.py
- data.py
- model.py
- preprocessing.py
- trainer.py
- utils.py
```

The previous structure have the following information:

* **tests:** This folder contain some Jupyter Notebooks in which I validated the correct behavior of the different methods developed in the scripts.
* **config:** This script contains the parameters values used through the differents .py files.
* **data:** This script is in charge of read and load the images, build the _input fn_ and define the data augmentation methods.
* **model:** This script defines model function, loss function and accuracy metrics needed to track the model performance.
* **preprocessing:** With this script we prepare the enviroment to before train the model.
* **trainer:** This files is in charge to train and validate the model defined.
* **utils:** This file contains some utility methods which are used by .py files.

## Model specifications and results

The architecture of this project are composed by two parts. The first one is a pretrained model and the second one is a custom architecture, both based on _Mobilnet_.

**So, Why I use a pretrained model?** Because, the pre-trained models have their parameters more adjusted and the filters they have learned are more polished, with which, we save the process of teaching the network those filters and the only thing that we should focus on is to adapt the other parameters to the specifications of our data set.

**How are the specifications of pre-trained model?** The pretrained model is an instance of MobilNet network, trained with imagenet weigths this architecture is provided by Keras API. For this model, I freeze all trainable parameters for get the best feature maps of my dataset.

**How is the architecture of custom model?** The custom model is a tiny mobilnet architecture, with 10 layers, one convolutional layer, one dense layer, four depthwise convolutional layers and four pointwise layers. All layer was trained with rectifier linear function and batch normalization before each activation, except the last layer which have softmax activation.

**How is the complete model trained?** You can see the complete list of hyperparameters that I use for train the model in the _config.yml_ file. The more relevant parameters are.
* batch_size: 10
* depth_multiplier: 1
* learning_rate: 0.001
* loss: 'sparse_categorical_crossentropy'
* num_classes: 10
* num_epochs: 45
* width_multiplier: 1
* optimizer: Adam

**Bonus** The full architecture of _Mobilnet_ (V1) is available in the _data.py_ file.If you want to try this network, you should change the following code.

* **_In trainer.py file_**:

Change this lines 

```
mobilnet_tiny = model.MobilNet_Architecture_Tiny(........
```
to 
```
mobilnet = model.MobilNet_Architecture(..........
```

and 

```
net = tf.keras.Sequential([
    base_model,
    mobilnet_tiny])
```
to
```
net = mobilnet
```

### Results
* **Training and Validation Accuracy**
![Training and Validation Accuracy](./pics/Training_Validation_Accuracy.png?raw=true)

* **Training and Validation Loss**
![Training and Validation Loss](./pics/Training_Validation_Loss.png?raw=true)

* **Training and Validation F1 Score**
![Training and Validation F1 Score](./pics/Epoch_F1_Training_Validation.png?raw=true)

* **Predictions in test set**
![Predictions in test set](./pics/Predictions_Test_Set.png?raw=true)


## How to use

This project are developed in Python 3 enviroment, is advisable install the following dependencies:

```
- Keras==2.2.4
- matplotlib==3.0.2
- numpy==1.16.1
- opencv-python==4.0.0.21
- scikit-learn==0.20.3
- scipy==1.2.1
- tensorflow-gpu==1.12.0
- tqdm==4.31.1
- urllib3==1.24.1
- wget==3.2
```

First of all we need to clone the repository.
```
git clone https://github.com/SebasPelaez/dog-breed-classification-mobilnet.git
```

Next, we move to the folder and execute the preprocessing script.
```
cd dog-breed-classification-mobilnet
python preprocessing.py
```

This script execute 4 functions.

1. **download_data:** Creates the data directory and extract there the compress file with the Standford Dog Dataset.
2. **extract_data:** Decompress the downloaded file.
3. **make_id_label_map:** Build a dictionary with the name of the dog's breed and a number representing the class, then save it in a .json file
4. **split_data:** Creates 3 different .txt files with the classes distribution. One file for training, one for validation and one for test the model.

At this point we have the enviroment ready to train the model.
```
python trainer.py -c config.yml
```

The previous steps allow us to train the model from the start, in the case that we want to train all parameters, but if we only want to use the predictor, we should download the pretrained weights from [here](https://www.dropbox.com/s/lfccfplsi0ry2rf/dog_breed_classification_mobilnet_checkpoints.rar?dl=1) and then extract the .rar file in the project root folder.

The project structure should be like that.

```
.
.
.
- tests/
    .
    .
    .
- checkpoints
- config.py
.
.
.
```

### How to use the predictor

Following the previous steps, we are ready to predict dog breeds. For do that we need to following this steps. 



**Notes:**

* Verify if your enviroment have all recommended dependencies.
* Change the config.yml to adapt the project parameters of your PC capacities.


## TODO

- [x] Create script for configure the enviroment.
- [x] Build Mobilnet architecture and _model fn_ function.
- [x] Create data managment and _input fn_ function.
- [x] Code wrap function to execute the train and test process.
- [x] Make Jupyter Notebooks with the unit test of some script functions.
- [x] Upload pretrained weights.
- [x] Show metrics and model results.
- [x] Create a Jupyter Notebook with the test of predictions script.
- [ ] Make predictions with different dog images.

https://www.dropbox.com/s/lfccfplsi0ry2rf/dog_breed_classification_mobilnet_checkpoints.rar?dl=0
_Feel free to make me comments, corrections or advices_