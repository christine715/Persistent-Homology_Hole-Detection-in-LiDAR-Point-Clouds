# Persistent Homology: Hole Detection in LiDAR Point Clouds
Suen Wun Ki Christine <br>
christinesuen715@gmail.com <br>
July 12, 2021 <br>
It is a computer vision algorithm used in my dissertation, which was submitted to the Hong Kong Polytechnic University in partial fulfillment of the requirement for the degree of Bachelor of Science (Honours) in Geomatics instructed by Dr. Wai Yeung Yan.
## Description
Missing area in the point clouds is one of the common problems in LiDAR. Therefore, the code for the detection of hole in LiDAR point clouds by Persistent Homology is presented.<br><br>
The main features: <br>
- Use of Persistent Homology in hole detection
- Apply Persistent Entropy to seperate topolgical noise and holes
- Classify type of hole by Maximum Likelihood Estimation

To apply Persistent Homology, Ripser.Py is used to detect the topological features in point clouds, which has an outstanding performance among various open-source PH libraries. Complete documentation about the package can be found at https://ripser.scikit-tda.org/en/latest/. The topological features is then seperated into holes and noises by Persistent Entropy, which works by  differentiating long from short lifetime of topological features. The output is the 3D coordinate of the points composed of the significant hole, hole's 3D area, longest distance between any two points of the hole and standard deviation of Z are output as file too. The hole classification is conducted based on the detection result. Training data have to be first built up and apply it into the likelihood model with the unknown hole. 
## Folder
Python script: 
1. Hole Detection
2. Hole Classification
## Installation
It is available on all major platforms. The libraries require to install include:
- numpy
- matplotlib
- ripser
- Cython
- persim
- math
- time
- ptlab
- open3d 
Below shows the example of commands typed into the environment to install ripser. 
```
pip install cython
pip install ripser
```
