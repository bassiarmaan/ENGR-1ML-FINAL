#  E96A Python and Machine Learning Final Project

## Overview:

Your final project will be to build and train a neural network classifier for the [MNIST handwritten digits dataset](https://pytorch.org/vision/stable/generated/torchvision.datasets.MNIST.html#torchvision.datasets.MNIST). Your classifier will be built in PyTorch using the modules we discussed in class and (optionally) other modules of your choice. You will write a short report on your classifier in the form of a slideshow presentation that you will present on the final day of class. After all presentations we will have a competition between classifiers in the class.

## MNIST Dataset

[The MNIST dataset](https://en.wikipedia.org/wiki/MNIST_database) is a collection of 70,000 handwritten digits, each of which is represented as 28x28 pixels in grayscale. Each digit also has a corresponding value from 0 to 9 (the tag of what the digit actually is). The following is documentation on how to load this dataset into your notebooks using PyTorch:  
[https://pytorch.org/vision/stable/generated/torchvision.datasets.MNIST.html\#torchvision.datasets.MNIST](https://pytorch.org/vision/stable/generated/torchvision.datasets.MNIST.html#torchvision.datasets.MNIST)

You are allowed to use the first 60,000 elements of this dataset however you like during training. It is recommended that you split the dataset into training, validation, and testing data, however this is up to you. You are also allowed to pre-process the data if you want, this is also up to you. Hint: [Data augmentation](https://nanonets.com/blog/data-augmentation-how-to-use-deep-learning-when-you-have-limited-data-part-2/) might be a good idea to make your model more robust to common image classification problems, such as rotated, inverted, or scaled images. On the final day we will be testing your classifiers on the MNIST dataset as well as digits that you will create.

## Classifier Requirements

Your classifier must be made using PyTorch. We will be testing your classifiers which means you need a way to get them to us – we will not be retraining them. You must make a class which extends the [torch.nn.Module class](https://pytorch.org/docs/stable/generated/torch.nn.Module.html) and then has all of the information about the layer sizes and types as well as the forward function. In addition you must [save your model weights and provide them to us](https://pytorch.org/tutorials/beginner/saving_loading_models.html). With these two things we will be able to recreate your trained model.

There are no requirements about your classifier’s training time – you may train it as much or as little as you want, you may choose when to stop training it, and you may do anything you want to train it. Your final model must be able to classify digits in a reasonable amount of time (\<1 second). This will likely not be an issue, most classifiers should be able to perform the classification in significantly less time.

## Competition

The competition will take place on the last day of class. During the competition all classifiers will be tested against the MNIST dataset. In addition each group will submit ten handwritten digits, one for each digit. We will use these handwritten digits to test the classifier’s accuracy as well.

The first part of the competition will be the test against the MNIST dataset. This will account for 50% of your score.

The second part of the competition will be individual tests against the handwritten digits you created. Each team will get tested on the same digit, so there is some strategy to this: your goal is to make digits that are recognizable enough for your classifier to predict, but poorly written enough that hopefully other group’s classifiers will not be able to predict them. Basically you are writing the worst digits you can that your classifier can still classify. Before the competition you will only be able to test/train your classifier using your own digits, not the digits of your classmates. 

Note: we will be examining your digits to be sure they are reasonable. This is because some teams may realize that some arbitrary collection of pixels is guessed correctly by their classifier just through luck and to make this fair we want to make sure that the digits are still valid digits.

Note 2: pre-processing your data may mean you can create interesting digits that will not work as well with groups that did not pre-process their data. We will accept digits that are rotated, flipped, and/or inverted as long as they still clearly match one digit (so 6s and 9s we may be a little more picky about).

## Presentation

Before the competition on the final day, each group will give a 5-10 minute presentation where they will talk about the design rationale for their neural network, what layers they used, how they trained it, any iterations they went through, and what their final testing accuracies were. Any other information you think is relevant to include in your presentation is up to you.

Note: EVERY group member must speak during the presentation.

## Submission

One submission per group. Same project score will be given to all the group members. You don’t need to let us know who is going to submit.

**What should I submit?**

1. Please package your jupyter notebook (.ipynb) for this project and submit it to BruinLearn. You should be able to find it under *Assignments-\>Final Project-\>Final Project Code(Jupyter Notebook)*. Do not delete code block outputs.  
2. Submit your final classifier as two files: one file named team0\_final\_project.py which contains the final project classifier class and one file named team0\_final.pth which contains the weights you used in your final classifier. In addition, please be sure that your classifier class is named ‘Classifier0’. In all cases of this paragraph replace 0 with your team number.  
3. Please submit your presentation powerpoints to BruinLearn. You should be able to find the submission link under *Assignments-\>Final Project-\>Final Project Presentation.*

**When should I submit?**  
The presentation powerpoints and the code are both due **December 2, 2024 at 11:59 pm**. Please start early because you **cannot** use your/your teammates’ late day passes on the project.

## Final Notes

This class, including the final project and competition are meant to be fun and low-stress. Please don’t stress out about this too much. If you complete the project you will be getting a good grade in the class, regardless of how well you do in the competition. Also, this is a popular classifier so you may be able to find versions online. While it will be tough to enforce this, we ask you to try to make your own rather than using an online version. It will make the competition more fun and fair, and ultimately is better for your learning (and again you don’t need to worry about your grade).   
