### Report on Classification Task

**Which method I used**

For this task, I decided that CNN should be a workable approach. However, because the dataset was very small (336 for the training set), it would be unfeasible to train an generalizable model from scratch. While this was an untested assumption, I decided that it would be more worth it to perform Transfer Learning. 

For transfer learning, I load the VGG model that has been pre-trained on ImageNet. I froze the weights of the Convolutional Layers and added a `global_average_pooling2d_2` in order to reduce variance and computation complexity. (In retrospect, I am not sure if this was a good choice) 

At the start of the experiment, I faced poor performance in the earlier few validation performance, but good performance in the training set - which impliex overfitting. Thus,  I iteratively tweaked the hyperparameters of the model to reduce overfitting. 

**Some of the effective changes to reduce overfitting made were as follows:** 

1. Reduce the number of neurons and layers to reduce architecture complexity 

2. Since ImageNet's initial dataset is quite different from the cancer cells images, I decided to train 2 convolutional layers instead of relying completely on the previous trained weights as feature extractor. This prove to give greater results 

3. Introduced dropout 

4. Introduced L2 Regularization 

   With that, I could see that while the `train` accuracy did not increase as quickly towards 1.0, the `validation` accuracy was gradually increasing, an encouraging sign. 



**Machine Learning Setup and Environment**

Initially for convenience's sake, I used my own Macbook's CPU for training, it took around 50 minutes to complete 10 epochs of training. I realised that to reduce overfitting, I had to experiment with the hyperparameters of the model faster. 

I set up my environment by uploading the dataset into my Google Drive and used Colab's library functions to mount the runtime environment onto the Cloud Drive, allowing me to access my files quickly. 

```python
from google.colab import drive
drive.mount('/content/drive')
```

In loading the images, I use some of Keras' utility helper functions that creates a Image Generator to generate images upon request. 

The `ImageDataGenerator` class has a method `flow_from_directory` that reads directly from the Cloud Drive's directory. 

**Final Outcome**

We manage to achieve a `validation_acc` of **0.7964**. However, there involve some level of cherry picking as we saved the checkpoint that performed best on the validation set. To be fair however, on the particular epoch that was saved, the model performed well on the `training_set` achieving **0.8307**. Maybe we can achieve a **fairer metric of accuracy** if we further split our entire data set to get the `test_set`.

Below is the classification report generated from `sklearn`. 

![./assets/screenshot.png)

 In this case, I am most interested in the `recall` of `positive` examples. In this classification task, there is high stakes in the prediction of `positive` cancer cells. A recall rate of 0.81 reflects that given a positive cancer cell, we were able to classify it correctly 81% of the time. Personally, I find that this begs improvement. 

We also have an `auc_score` of `0.7994538834951456` I think this is decent, but still begs improvement. 

####What I can do better in the future 

In general I believe that classification of **Cancer Cells did not benefit as much from transfer learning**. Imagenet and  other pretrained weights were trained on everyday objects, and thus learnt to extract relevant features that might not be common or beneficial to Cancer Cells. 

**On Average Pooling**

I believe that even for the CNN algorithm, the cancer cells is not as easily distinguishable as other common binary classification CV tasks such as Dogs vs Cats. This is because the cancer cells are probably distinguished on very granular features. Intuitively, I believe average pooling makes the image "**blur-er**". While it helps to reduce computation cost for easily distinguishable visual patterns, I am not sure if it was a good approach for distinguishing cancer cells. 

**On Data Imbalance**

The data imbalance is actually quite insignifcant with the positive classes taking only 35% of the total number of examples. I should have spent some time to balance the data by undersampling the negative examples. Another technique could be to apply data augmentation to a greater extent to the **positive** examples while doing less for the **negative** examples. 

| +/-      | Train | Validation |
| -------- | ----- | ---------- |
| Positive | 117   | 64         |
| Negative | 219   | 103        |

