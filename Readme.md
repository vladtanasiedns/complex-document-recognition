# Complex document processing

The script uses the keras framework for python to train and test on the [RVL-CDIP Dataset](http://www.cs.cmu.edu/~aharley/rvl-cdip/)
The dataset contains 400,000 grayscale images in 16 classes, with 25,000 images per class. 320,000 training images, 40,000 validation images, 40,000 test images. The images do not exceed 1000 pixels and are saved in ``` .tif ``` format.

## Document categories
* 0 letter
* 1 form
* 2 email
* 3 handwritten
* 4 advertisement
* 5 scientific report
* 6 scientific publication
* 7 specification
* 8 file folder
* 9 news article
* 10 budget
* 11 invoice
* 12 presentation
* 13 questionnaire
* 14 resume
* 15 memo

A. W. Harley, A. Ufkes, K. G. Derpanis, "Evaluation of Deep Convolutional Nets for Document Image Classification and Retrieval," in ICDAR, 2015