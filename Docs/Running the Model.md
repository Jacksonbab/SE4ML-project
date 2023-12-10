
# Running the Model 
1. Go to /home/husarion/Main_docker folder run the following.
   ```
   ./start.sh
   ```
2. In another terminal in the same directory once that has finished. It may take a while. 
   ```
    ./start_camera_node.sh    
   ```
   If this script return error, it means the camera is busy; try run ``./kill.sh`` or check ./kill.sh 's content to see how to kill ongoing process that occupies the camera. 
   Alternatively ``run sudo chmod 777 -R /dev/video0 `` if it is a permission issue

3. Running 
   Before you run the mode make sure that the robot is is suspended or held so it does not drive off a table. 
   In /home/husarion/ros2_tutorial_workspace run the following command in another terminal. 
   ```
   ./run_inference.sh
   ```
4. Once the model is running, disconnect all attached wires and place the robot on your chosen track

5. To stop the model reconnect the keyboard and press CTRL+C 
