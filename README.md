# Introduction to Machine Learning with Tensorflow Nanodegree
# Deep Learning
## Project: Image-Classifier

### Install

This project requires **Python 3.7** and the following Python libraries installed:

- NumPy
- matplotlib
- Tensorflow
- Tensorflow Hub
- Tensorflow Datasets
- JSON
- PIL

You will also need to have software installed to run and execute an [iPython Notebook](http://ipython.org/notebook.html)

### Problem Statement

A Primitive AI Application as an image classifier to recognize different species of flowers. Similar to a phone app that tells you the name of the flower your camera is looking at. Training the classifier, then exporting it for using it in the application. 

### Data

102 category dataset, consisting of 102 flower categories. The flowers chosen to be flower commonly occuring in the United Kingdom. The dataset is found at: http://www.robots.ox.ac.uk/~vgg/data/flowers/102/index.html

### Description

- udacitu_project.h5 is the frozen graph.
- Image Classifier - Part 1 - Development.ipynb involves Development of the Image Classifier with Deep Learning.
- predict.py is Command Line AI Image Classifier Application.
- Command Line Arguments:

Basic usage:
  -- $ python predict.py /path/to/image saved_model

Options:

    --top_k : Return the top KKK most likely classes:
    $ python predict.py /path/to/image saved_model --top_k KKK

    --category_names : Path to a JSON file mapping labels to flower names:
    $ python predict.py /path/to/image saved_model --category_names map.json

