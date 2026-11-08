---
title: Homework 2 -- Multiple Object Video Detection
author: Oksana Moskviak
course: Artificial Intelligence · UCU
date: 2026-10-12
---

## Introduction
In this homework, I had to explore training Deep Learning models for multiple object video detection task on drone videos from the VisDrone2019-VID benchmark. I experimented with 4 types of Computer Vision models: Two-stage Anchor-based (Faster R-CNN ResNet-50 FPN), One-stage Anchor-based (SSD VGG-16), One-stage Anchor-free (YOLO26n) and Query-based (RT-DETR).

## Method

### Data

#### Data Understanding
For data understanding, I have explored the number of clips for each dataset (train, validate, test) along with the the max height and width, min height and width of the images, number of ignored boxes, a check of shared videos, plotted class distribution of the data, area of bounding box along with width and height of boxes.

![Training loss](images/classes-distribution.png)
The data is highly imbalanced - cars are the most frequent class in training, while people class dominate validation. 

![Width and height of bounding boxes](images/width-height.png)
![Area of bounding boxes](images/area.png)

Many bounding boxes are only tens of pixels wide and high, which means that reducing image size may make these objects harder to detect. This is also supported by the plot of the area of the bounding boxes. 

![Objects per frame](images/objects.png)
Most training frames contain fewer than 50 objects, but some have more than 150, which makes detection in crowded scenes challenging. Validation frames generally contain fewer objects, which means that the model is evaluated on less crowded scenes. 

#### Data Preprocessing
For data augmentation, I applied random horizontal and vertical flips, each with a probability of 50%, and adjusted the changes to the bounding boxes accordingly. These transformations are applied in order to give the model more examples of objects in different orientations. They were only applied to the training data, validation images were unchanged. 
I converted these images from BGR to RGB and transformed them into tensors for the further training. The change of image size was applied according to the models requirements.

### Training

#### Settings
I applied cross-validation to 3 models: the two-stage, one-stage anchor-based and one-stage anchor-free detectors, to find the best hyperparameters for each. The validation folds were created from the training dataset, using every 15th frame. After this search, I trained each final model with its best parameters combination, using every 5th frame from the training dataset. I made this choice after inspecting the images and noticing that the frames were mostly similar, with the same objects still present after 5 frames. The reason for sampling fewer frames is to reduce the training and validation time and computational costs.


####  Two-stage Anchor-based model



## Results

| Model    | Accuracy | F1   |
|----------|----------|------|
| Baseline | 0.71     | 0.68 |
| Ours     | 0.84     | 0.82 |

Images go into the `images/` folder:

<!-- ![Training loss](images/loss.png) -->

## Conclusion

> Summarize your findings here.
