# How to collect Data
1. Use 'Bluetoothctl' command in linux, then using 'scan on' command to find the MAC address of the Xbox Controller
   Use 'pair <MAC address>' or use 'connect <Mac addresss>' to connect the Xbox Controller. Press both connection buttons on the Xbox Controller if it does not appear.


2. Go to /home/husarion/Main_docker folder run the following.
   ```
   ./start.sh
   ```
3 In another terminal in the same directory once that has finished. It may take a while. 
  ```
   ./start_camera_node.sh    
  ```
  If this script return error, it means the camera is busy; try run ``./kill.sh`` or check ./kill.sh 's content to see how to kill ongoing process that occupies the camera. 
  Alternatively ``run sudo chmod 777 -R /dev/video0 `` if it is a permission issue

4. In /home/husarion/ros2_tutorial_workspace run the following command in another terminal. 
   ```
   ./run.sh
   ```
# Collection Prodcedure 
Once you have run ``./run.sh``
press Y on gamepad to start collection.
press A when you finish collection.
!!! You should hold and press the buttons for a couple seconds or press multiple times to make sure it works. 

You can collect multiple times as long as the code is running. Ensure you finish the previous collection cleanly by pressing A multiple times.
When you press A to finish collection. Your data will be written into those three files inside the home/husarion/Data folder

Disconnect all external plugs such as the charger or HDMI from the robot and make sure that it is stable before trying to drive it around. 

To drive the robot you must hold down the right bumper and then use the stick to control acceleration. 
