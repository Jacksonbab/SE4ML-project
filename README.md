# SE4ML-project

# Contents 
Data collection and Robot Running Instructions are located in the **docs** folder 

Model Training Instructions are inside **Model_Training** folder

# Installation 
Follow basic rosbot setup guide https://husarion.com/tutorials/howtostart/rosbotxl-quick-start/

You should have a camera or lidar to collect the data. The type of camera and lidar satisfied for the RosbotXL is also in the husarion.com website.

/home/husarion/Main_docker contains all the relevent docker images for this project. The run commands to use them are in the next section. 
Install any prerequisites that may come up when running those commands. 

# Model Evaluation
Our robot can sometimes recognize the wall and the stairs of the rice hall 4th floor, but sometimes it's going to be hesitate with whether it should turn left or right in an open area. This is a very common phenomenon since our model do not have collect enough data around the open area. Also, since we do not collect every turning angle of robot, so it may sometimes get straight into the wall with only linear speed.
