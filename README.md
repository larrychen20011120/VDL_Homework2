# VDL_Homework1
## Introduction
This is the HW2 in visual deep learning. In this project, we should predict the bounding boxes and categories of digits in the given image and then output the whole number of this image. This could be viewed as two tasks. I apply Faster RCNN with pretrained backbone of MobileNet-V3 on `torchvision` which is called `FasterRCNN_MobileNet_V3_Large_FPN_Weights.COCO_V1`. The following table shows the hyper-parameters for our training:


## Project structure
- `train.py` is the main function for training
- `model.py` is the file describe the Faster RCNN
- `data.py` describes how to build up the pytorch dataset and the processing methods and augmentation methods I used
- `ensemble.py` is to ensemble many model's outputs together
- `inference.py` is to run testing result
- `find_threshold.py` find the best threshold of score under the validation set
- other python files are from torchvision which is the easy training code I can simply apply to this task

## How to run the code
- install the dependencies and activate the environment
  ```
  conda env create --file=environment.yaml
  conda activate DL-Image
  ```
- Generate the sample data augmentation (stored as `bbox.png`)
  ```
  python dataset.py
  ```
- See the model size for training
  ```
  python model.py
  ```
- train the model (if use default parameter, just run the following code). You can change the `LOG` name in `engine.py` to alter the tensorboard log filename
  ```
  python train.py
  ```

## Performance


### Reference
- [Kaggle Tutorial of Fine-tuning Faster-RCNN Using Pytorch](https://www.kaggle.com/code/yerramvarun/fine-tuning-faster-rcnn-using-pytorch#Model)
- [TorchVision Object Detection Finetuning Tutorial](https://pytorch.org/tutorials/intermediate/torchvision_tutorial.html)