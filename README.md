# Semantic Segmentation as a Service
## A project that aims at building a Semantic Image Segmentation model and depoying it as a service.

### Introduction
Semantic image segmentation is the task of assigning a semantic label, such as “road”, “sky”, “person”, “dog”, to every pixel in an image. 
This project is based on Google's [Deeplab](https://github.com/tensorflow/models/tree/master/research/deeplab) project which has also enabled numerous new applications, such as the synthetic shallow depth-of-field effect shipped in the portrait mode of the Pixel 2 and Pixel 2 XL smartphones and mobile real-time video segmentation-[source](https://ai.googleblog.com/2018/03/semantic-image-segmentation-with.html).
The dataset used here is a well known autonomous drving dataset called [CityScapes](https://www.cityscapes-dataset.com/)

![alt text](https://github.com/pratikkulkarni228/segmentation-as-a-service/blob/master/figs/deeplabv3.png)
[source](https://arxiv.org/pdf/1802.02611.pdf)

### Frameworks/ tools used:
1. Google DeepLab
2. Tensorflow
3. OpenCv
4. Flask 
### Service:
![alt text](https://github.com/pratikkulkarni228/segmentation-as-a-service/blob/master/figs/home.png)
![alt text](https://github.com/pratikkulkarni228/segmentation-as-a-service/blob/master/figs/pred.png)
# How to setup the service:
1. Clone this repository.
2. [IMPORTANT] Download the trained, converted and zipped model from [this link](https://drive.google.com/open?id=111lkKq_EvvpVut-V3oGaGbbHEWTowRQ2).
3. [IMPORTANT] Place it in the cloned repo directory
4. [OPTIONAL] Create a virtual env   
Using CONDA:  
``conda create -n yourenvname python=3.6 anaconda``  
``source activate yourenvname``  
OR  
Using PIP:  
``pip3 install virtualenv``  
``virtualenv myenv``  
``source mypython/bin/activate``  
OR  
WINDOWS:  
``virtualenv myenv``  
``.\myenv\Scripts\activate``  
5. [IMPORTANT] Run the following command (DO NOT SKIP THIS STEP):  
``pip3 install -r requirements.txt``
6. Once installed, run the following command to launch service:  
``python3 app.py``  
If you encounter OS error or Permission error, You might have to change the permissions of the ``app.py`` file as follows:  
``chmod 644 app.py ``
7. Open http://127.0.0.1:5001/ to access the service.
8. Happy Segmenting

### Process Pipeline:  
![alt text](https://github.com/pratikkulkarni228/segmentation-as-a-service/blob/master/figs/seg-proc.png)

### Steps involving entire training process:
1. Register and Download the following files from cityscapes dataset website. leftimg gtfine
``leftImg8bit_trainvaltest.zip (11GB) [md5]`` and ``gtFine_trainvaltest.zip (241MB) [md5]``
2. Clone the deeplab [respository](https://github.com/tensorflow/models) and follow the further instructions from [here](https://github.com/tensorflow/models/tree/master/research/deeplab).
3. Convert your dataset to tfrecord format that is used by tensorflow, from [here](https://github.com/tensorflow/models/blob/master/research/deeplab/g3doc/cityscapes.md)
4. Download the model checkpoint from [here](https://github.com/tensorflow/models/blob/master/research/deeplab/g3doc/model_zoo.md) (This serves as an initial model checkpoint which will be used to train further)
5. Train, evaluate and visualise your model results by following the instructions [here](https://github.com/tensorflow/models/blob/master/research/deeplab/g3doc/cityscapes.md).
6. Convert the checkpoint files to a frozen graph (.pb) format for easy inference. (use export_model.py in models/research/deeplab/) and convert the .pb to .tar.gz
7. For Inference (single image inference) run the inference.py file to generate segmented outputs

