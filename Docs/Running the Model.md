
# Running the Model 
Start all nodes:
go to /home/husarion/Main_docker folder run the following.
```
./start.sh
```
And in another terminal in the same directory once that has finished. It may take a while. 
```
 ./start_camera_node.sh    
```
If this script return error, it means the camera is busy; try run ``./kill.sh`` or check ./kill.sh 's content to see how to kill ongoing process that occupies the camera. 
Alternatively ``run sudo chmod 777 -R /dev/video0 `` if it is a permission issue

In /home/husarion/ros2_tutorial_workspace run the following command in another terminal. 
```
./run_inference.sh
```
