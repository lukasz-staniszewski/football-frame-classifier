<h1 align="center">Footbal Frames Classifier</h1>
<h2 align="center">Łukasz Staniszewski</h2>

<br>
<div align="center">
<img src="https://user-images.githubusercontent.com/59453698/170739138-44ee7ba7-e7b2-456b-90bb-e4d5a7c741d6.png" alt="banner">
</div>

<h2 align="center"> I. Task description </h2>
Aim of the task is to create a model, which will differentiate between different types of views encountered during a football match. We would like to be able to select only those frames, which contain valuable information to be processed by our data extraction system. The task is stated as a multiclass classification problem.

<h2 align="center"> II. Provided data </h2>

Two datasets are provided:
+ train (10987 images)
+ test (3117 images)

Train dataset should be used for model training and validation, whereas the test dataset should only be used for running the final prediction part. Predictions from the test dataset should be attached in your solution, according to the Task solution description.

Labels are thus provided only for the train dataset - df_train_framefilter.csv. The csv file contains two columns:
+ basename: filename of an image
+ category: label of an image

The following classes are present in the dataset:
+ side_view
+ closeup
+ non_match
+ front_view
+ side_gate_view
+ aerial_view
+ wide_view

Filenames have a certain structure, ex.:
```
1e2251a894690ac79324b119-00000421.jpg
```
+ contains match_hash-frame_index, where:
    + match_hash: hashed match name
    + frame_index: global frame index from a video


<h2 align="center"> III. First observations </h2>
After first anaylsis made with given dataset, there are two factors, which make this task more difficult:

+ class imbalance - after analysis of csv file, huge variations in given class samples have been noticed:

```python
side_view         6200
closeup           2812
non_match         1638
front_view         114
side_gate_view     109
aerial_view         67
wide_view           47
Name: category, dtype: int64
```

+ difficult class differentation - there can be huge problem for nn to differ images of classes side_view, front_view, side_gate_view, aerial_view, wide_view - to be honest, even if I was trying, I couldn't guess the true class of the picture - reason for this is a very high similarity of the images of these classes, which, combined with a high imbalance, may be an obstacle.

To fight these two problem, I decided that:

+ I will use class weightning for Cross Entropy Loss function (which will hopefully deal with imbalance).
+ because aim of task is to filter only those frames, which contain valuable information to be processed, the most important metrics for validation are good accuracy, precision and F1 score for classes close_up, non_match and one of the *_view and not bad metrics for remaining classes.
+ While validation I decided to count after each epoch metrics of accuracy, precision, recall and F1 in both micro and macro versions - macro versions of these metrics are more sensitive to classes of smaller size, what can help in choosing model.

<h2 align="center"> IV. Data preprocessing </h2>
Only preprocessing which was performed in this solution is permanent resize of given images to size 360x640 which kept the 16:9 aspect ratio but helped in faster data loading.

To perform this preprocessing you need to use python script:
```sh
$ python preprocess_data.py -i FOLDER_DATA_TRAIN -o FOLDER_DATA_TRAIN_RESIZED

$ python preprocess_data.py -i FOLDER_DATA_TEST -o FOLDER_DATA_TEST_RESIZED
```

Also, there was a need to count means and stdds for each channel in training data to make input images normalized - this process was made in notebooks/DataAnalysis.ipynb jupyter notebook.

Also, prepared project performs training-validation split.

<h2 align="center"> IV. Model and training </h2>
As model I decided to use ready architecture - ResNet-18 with additional Fully Connected layer which helps return probabilities of all 7 classes in this task.

As loss function, Cross Entropy Loss has been used with with class weightning, which help with fighitng with class imbalance.

The best, regarding to assumptions made about the metrics, became model with data normalization, and class weightning.

To train model, you have to fill config.json file with specific elements:

```json
...
    "arch": {
        "type": "ResNetClassifier",
        "args": {
            "num_classes": 7
        }
    },
    "data_loader": {
        "type": "FramesDataLoader",
        "args": {
            "images_folder": "FOLDER_DATA_TRAIN_RESIZED",
            "batch_size": 128,
            "is_with_aug": false,
            "csv_path": "data/df_train_framefilter.csv",
            "csv_path_tf": "data/df_train_framefilter_rs.csv",
            "shuffle": true,
            "validation_split": 0.2,
            "num_workers": 2
        }
    },
    "optimizer": {
        "type": "Adam",
        "args": {
            "lr": 0.00001,
            "weight_decay": 0,
            "amsgrad": true
        }
    },
    "loss": "cross_entropy_loss",
    "metrics": [
        "micro_accuracy",
        "micro_recall",
        "micro_precision",
        "micro_f1",
        "macro_accuracy",
        "macro_recall",
        "macro_precision",
        "macro_f1"
    ],
    "lr_scheduler": {
        "type": "StepLR",
        "args": {
            "step_size": 50,
            "gamma": 0.1
        }
    },
    "trainer": {
        "epochs": 100,
        "save_dir": "saved/",
        "save_period": 1,
        "verbosity": 2,
        "monitor": "min val_loss",
        "early_stop": 10,
        "tensorboard": false
    },
    "class_weights": [
        0.253,
        0.558,
        0.958,
        13.768,
        14.400,
        23.426,
        33.395
    ],
...
```
And then run python script performing training:

```sh
$ python train.py -c config.json
```

<h2 align="center"> V. Validation </h2>
From training logs, you can read validation statistics and model location path, for example:

```
saved/models/FootballFramesClassifier/0527_015601/best_model.pth
```

You can use it to perform validation once more. To do it, use python script:
```
$ python evaluate_metrics_val.py
```  

Results of best model in this project:

```python
~~PER CLASS ACCURACIES~~
Accuracy for side_view = 98.94%
Accuracy for closeup = 91.26%
Accuracy for non_match = 77.19%
Accuracy for front_view = 73.68%
Accuracy for side_gate_view = 76.19%
Accuracy for aerial_view = 66.67%
Accuracy for wide_view = 85.71%
~~~~~~~~~~~~~~~~~~~~~~~~
~~~~PREDICTIONS MADE:~~~
side_view         1227
closeup            612
non_match          289
front_view          23
side_gate_view      23
aerial_view         13
wide_view           10
dtype: int64
~~~~~~~~~~~~~~~~~~~~~~~~
~CLASSIFICATION REPORT:~
                precision    recall  f1-score   support

     side_view       0.99      0.99      0.99      1229
       closeup       0.89      0.91      0.90       595
     non_match       0.85      0.77      0.81       320
    front_view       0.61      0.74      0.67        19
side_gate_view       0.70      0.76      0.73        21
   aerial_view       0.31      0.67      0.42         6
     wide_view       0.60      0.86      0.71         7

      accuracy                           0.93      2197
     macro avg       0.71      0.81      0.75      2197
  weighted avg       0.93      0.93      0.93      2197

~~~~~~~~~~~~~~~~~~~~~~~~
```
As we can clearly see, model is working really good for classes side_view, closeup, non_match in terms of metrics precision, accuracy and f1-score, what was wanted to be achieved.

On the other hand, there is still bad performance of model for class aerial_view and wide_view - as stated before, this is probably due to the:
+ similarity of frames for this class and other *_view classes
+ high imbalance in provided dataset with negative impact for these classes.

The way to deal with it is to prepare less imbalanced dataset for model. Maybe architecture filtering 'view' vs 'no_view' classes with additional architecture distinguishing between 'view' class frames could perform well.

<h2 align="center"> VI. Predictions </h2>
To get final predictions of model you need to use python script:

```python
$ python make_predictions.py
```

Running it on our model creates 'results.csv' file with all necessary data and prints results of predictions:

```python
Predictions distribution:
side_view         1682
closeup            889
non_match          427
front_view          50
side_gate_view      45
aerial_view         13
wide_view           11
Name: category, dtype: int64
Predictions performed to file: results.csv
```

As we can clearly see, class distributions of predictions is similiar to distribution of provided dataset.

<h2 align="center"> VII. Conclusion </h2>
During this project I found out that to achieve good results in views classification, there is a necessity to prepare good data and preprocess it in correct way. I have also found out that there is no need to use Full-HD images for model and resizing them may safe much time. Also, using class weights is a good way to deal with dataset imbalance. 

<h2 align="center"> VII. Additional information </h2>

+ All experiments are reproducible thanks to random seeds.
+ Project was made using google colab.
+ In local development, Python 3.9.2 was used with Anaconda, all necessary modules are in requirements.txt.
+ Folder structure:
  ```
  footbal-frame-classifier/
  │
  ├── results.csv - final predictions
  ├── train.py - main script to start training
  ├── make_predictions.py - script for making predictions
  ├── preprocess_data.py - script for data preprocessing
  ├── evaluate_metrics_val.py - script for model validation  
  │
  ├── requirements.txt - necessary modules to develop locally
  ├── config.json - holds configuration for training
  ├── parse_config.py - class to handle config file and cli options
  │
  ├── notebooks/ - notebooks used in project
  │   ├── DataAnalysis.ipynb - notebook for data preprocessing
  │   └── Colab.ipynb - Google Colab session using all scripts
  │
  ├── base/ - abstract base classes
  │   ├── base_data_loader.py
  │   ├── base_model.py
  │   └── base_trainer.py
  │
  ├── data_loader/ - anything about data loading goes here
  │   ├── TestDataset.py    - dataset for tests
  │   ├── FramesDataset.py  - dataset for train/validation
  │   └── data_loaders.py
  │
  ├── data/ - default directory for storing input data
  │
  ├── model/ - models, losses, and metrics defined
  │   ├── model.py
  │   ├── metric.py
  │   └── loss.py
  │
  ├── saved/
  │   ├── models/ - trained models are saved here
  │   └── log/ - default logdir for tensorboard and logging output
  │
  ├── trainer/ - trainers
  │   └── trainer.py
  │
  ├── logger/ - module for tensorboard visualization and logging - not used in this project
  │  
  └── utils/ - small utility functions
      ├── util.py
      └── ...
  ```


## Author
Project was created by <a href="https://github.com/lukasz-staniszewski">Łukasz Staniszewski</a> in order to solve recruitment task. 

Used PyTorch projects template - <a href="https://github.com/victoresque/pytorch-template">victoresque/pytorch-template</a>.
