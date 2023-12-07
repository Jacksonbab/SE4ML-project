# SE4ML-project
# How to collect data and how to run robot:
Use 'Bluetoothctl' command in linux, then using 'scan on' command to find the MAC address of the Xbox Controller
Use 'pair <MAC address>' or use 'connect <Mac addresss>' to connect the Xbox Controller

start all nodes:
go to /home/husarion/Main_docker folder run 
'''
./start.sh
'''
then  ./start_camera_node.sh    (if this script return error, it means the camera is busy; try run ./kill.sh or check ./kill.sh 's content to see how to kill ongoing progress that occupies the camera)
Need to wait sometime for the Xbox Controller setup

run collection code
go to ros2_tutorial_workspace 
run  ./run.sh

# start collection
press Y on gamepad to start collection.
press A when you finish collection.
!!! need to press Y or A multiple times in case the collection process not start or end.

You can collect multiple times as long as the code is running. Just
ensure you finish the previous collection cleanly by pressing A multiple times.
When you press A to finish collection. Your data will be written into those three files inside the home/husarion/Data folder
