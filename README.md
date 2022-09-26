# Ocular Disease Recognition

This project builds a custom network in TensorFlow for classifying diseases by using ocular scan data.

## Description

By leveraging Kera's functional architecture we can specify and compile a custom model for identifying one of 8 possible
diagnoses when presented with a patients ocular scans. This will allow faster, more accurate, and more convenient diagnosis
for many common issues that are traditionally found manually during an optometrist's review.

## Getting Started
Be sure to download the dataset and extract it into the 'Data' folder:

https://www.kaggle.com/datasets/andrewmvd/ocular-disease-recognition-odir5k?resource=download


### Dependencies
* Cv2: 4.6.0
* TensorFlow / Keras: 2.10.0
* Pandas: 1.4.4
* Numpy: 1.21.5
* imutils: 0.5.4

### Installing and Running

* Clone repo
* Run ```python3 O_D_Recognition.py --data_path path/to/images --label_path path/to/labels.xlsx```

### Help
For errors with GCC version ( AKA GLIBCXX errors from Pandas )
```
sudo add-apt-repository ppa:ubuntu-toolchain-r/test # Ignore if not ubuntu

sudo apt-get update

sudo apt-get install gcc-4.9

sudo apt-get upgrade libstdc++6
```
#### Verify the required version is now present:
```
strings /usr/lib/x86_64-linux-gnu/libstdc++.so.6 | grep GLIBCXX
```