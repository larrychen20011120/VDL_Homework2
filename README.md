# VDL_Homework1
## Introduction
This is the HW2 in visual deep learning. In this project, we should predict the bounding boxes and categories of digits in the given image and then output the whole number of this image. This could be viewed as two tasks. I apply Faster RCNN with pretrained backbone of MobileNet-V3 on `torchvision` which is called `FasterRCNN_MobileNet_V3_Large_FPN_Weights.COCO_V1`. I set all parameters of the Faster RCNN to be trainbale and apply different learning rate to different layers of the model. I choose Adam as the optimizer and apply cosine annealing learning decay. The detailed hyper-parameters for training is shown as the following:

![image](https://github.com/user-attachments/assets/f0e416fe-0115-4dc2-9829-d934fa96c057)

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
- Find the best threshold for Task 2
  ```
  python find_threshold.py
  ```
- Inference (You can setup your `model weights` and `threshold`)
  ```
  python inference.py
  ```
- Ensemble Multiple models' results
  ```
  python ensemble.py
  ```

## Performance
The validation score of my method.
![螢幕擷取畫面 2025-04-16 223059](https://github.com/user-attachments/assets/cc41329f-3b70-4397-9cd4-7ee114472ef2)



### Reference
- [Kaggle Tutorial of Fine-tuning Faster-RCNN Using Pytorch](https://www.kaggle.com/code/yerramvarun/fine-tuning-faster-rcnn-using-pytorch#Model)
- [TorchVision Object Detection Finetuning Tutorial](https://pytorch.org/tutorials/intermediate/torchvision_tutorial.html)
