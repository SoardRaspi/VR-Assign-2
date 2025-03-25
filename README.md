# VR-Assign-2

## Task 1
Binary Classification Using Handcrafted Features and ML Classifiers. This task mainly aims at the binary classification of an input image into whether the person in the image is wearing a mask. The dataset used for this task is from [this]([https://github.com](https://github.com/chandrikadeb7/Face-Mask-Detection/tree/master/dataset)) GitHub repo. The dataset contains two types of images, stored in two folders with the respective names; images of people with mask and without mask.

### 1.1 Extracting handcrafted features
For this task, we consider two types of features; Histogram of Oriented Gradients (HOG) and Histogram of Canny Edge detection features.

* The HOG features for the input images are extracted using the scikit-learn’s in-built function. This captures local edge patterns by computing gradient orientations in small regions of an image. For the task, normalization was done considering 2x2 cells together. Each cell is a block of 8x8 pixels. For calculating the gradients, each gradient is divided into 9 bins between 0 and 180 degrees. This method captures edge and texture details which is used for classification of the imaged using different models.
* The Canny edge features for the input images are extracted using OpenCV’s built-in Canny edge detection function. This method identifies edges by detecting intensity changes and suppressing weaker ones to highlight only the strongest edges. For this task, the edge detection thresholds were set to 50 and 150. Once the edges are detected, a histogram of pixel intensities is generated with 256 bins, covering values from 0 to 255. The histogram is then normalized to represent the distribution of edge intensities. These features help capture the prominence of edges in the image and are used for classification with different models.

### 1.2 Training and Evaluating models

## Exceution Steps
### Task2
- Run the cell containing ```model = models.load_model("/content/mask_detection_model_small.h5")``` only to load an existing model and replace the argument with the required path.
- Run the rest as usual

### Task3
The code was ran on google colab, to run it on a local system do the following:
-  Comment out imports with  ```google.colab``` and ```drive.mount(...)``` and ignore the 'zipfile' cell
-  Instead of ```train_folder = '/content/MSFD/1/face_crop'``` and   ```
test_folder = '/content/MSFD/1/face_crop_segmentation'``` replace them with our own paths for face_crop and face_crop_segmentation folders and run the rest
of the cells as usual.
To run it on Colab, replace ```zip_path = '/content/drive/MyDrive/Colab Notebooks/MSFD.zip'``` with the actual path to the zip file in your drive

### Task4
The folder structure is as follows:

├── imageSearch <br/>
│   ├── config.py <br/>
│   ├── dataset.py <br/>
│   └── model.py <br/>
├── predict.py <br/>
└── train.py <br/>

``` model.py``` contains the actual model architecture
``` dataset.py``` contains contains the torch dataset class of the given data
```config.py``` contains all the global paths and parameters like dataset path (image dataset path, mask dataset path), model path, test and plot paths, test split, initial learning rate etc.
  
- In ```config.py``` set ```DATASET_PATH = "<your-dataset-path>"```
- Optionally to modify the output paths or hyperparameters of the model modify it accordingly in the ```config.py``` file.
- To train the model, run: ```python3 train.py```. The model will then be saved in the ```MODEL_PATH``` specified in the config file along with plots in the ```PLOT_PATH```.
- To perform predictions, run: ```python3 predict.py```.
