# SE4ML-project

# Contents 
Data collection and Robot Running Instructions are located in the **docs** folder 

Model Training Instructions are inside **Model_Training** folder

# Installation 
Follow basic rosbot setup guide https://husarion.com/tutorials/howtostart/rosbotxl-quick-start/
You should have a camera or lidar to collect the data. The type of camera and lidar satisfied for the RosbotXL is also in the husarion.com website. This project used a zed 2 camera and did not use the lidar. 

/home/husarion/Main_docker contains all the relevent docker images for this project. 
The commands to use them are in Docs Folder. Running the model and data collection is aslo in the docs folder.
Install any prerequisites that may come up when running those commands. 

# Resources
Model checkpoints can be downloaded [here](https://drive.google.com/drive/folders/11L-rNwJL3N83MnqB6JUFCWAoIH0nj-zN?usp=sharing)

Datasets used for training can be downloaded [here](https://drive.google.com/drive/folders/11L-rNwJL3N83MnqB6JUFCWAoIH0nj-zN?usp=sharing)
# Model Evaluation

Our robot can sometimes recognize the wall and the stairs of the rice hall 4th floor, but sometimes it's going to hesitate with whether it should turn left or right in an open area. This is a very common phenomenon since our model do not have collect enough data around the open area. Also, since we do not collect every turning angle of robot, so it may sometimes get straight into the wall with only linear speed. 

Performance also depends on which model is deployed as several models were made. Models that were trained on driving perfectly did not react to walls, but could sometimes make turns well. Models trained on the snake traversal data did react to walls, but sometimes would turn into rather than away from wall. Snake trained data also would try to make turns, but the reaction was often too strong that it would turn almost completely around instead of making the turn. 
