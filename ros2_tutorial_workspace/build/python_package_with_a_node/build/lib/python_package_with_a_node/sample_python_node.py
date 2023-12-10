#!/usr/bin/env python
from datetime import datetime
import os
import random
import string
import time
import numpy as np
import cv2
from cv_bridge import CvBridge
import threading
import json

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, BatteryState, LaserScan, Range, Joy
from geometry_msgs.msg import Twist

# globals
image = None
speed_cmd = 0.0
turn_cmd = 0.0
collecting = False
batt_state = vel_state = lidar_state = None
range_fl = range_rl = range_fr = range_rr = None    

class ImWriteThread(threading.Thread):
    def __init__(self, dataset_subdir):
        super(ImWriteThread, self).__init__()
        self.im = None
        self.im_timestamp = None
        self.img_filename = ""
        self.vel_lin_x = None
        # adding other lin and ang values here
        self.vel_lin_y = None
        self.vel_lin_z = None
        self.vel_ang_x = None
        self.vel_ang_y = None
        self.vel_ang_z = None
        self.joy_stick_data = None
        
        self.dataset_subdir = dataset_subdir
        self.img_count = 0
        self.timeout = None
        self.done = False
        self.condition = threading.Condition()
        self.start()

    def run(self):
        while not self.done:
            self.condition.acquire()
            self.condition.wait(self.timeout)
            if self.im is not None:
                self.img_filename = "rosxl-{:05d}.jpg".format(self.img_count)
                cv2.imwrite("{}/{}".format(self.dataset_subdir, self.img_filename), self.im)
                with open(self.dataset_subdir + "_data.csv", 'a') as f:
                    f.write("{},{},{},{},{},{},{},{},{}\n".format(self.img_filename, self.im_timestamp, datetime.now(), self.vel_lin_x, 
                    # adding other lin and ang values here
                    self.vel_lin_y,
                    self.vel_lin_z,
                    self.vel_ang_x,
                    self.vel_ang_y,         
                    self.vel_ang_z))

                
                joy_buttons = self.joy_stick_data.buttons
                joy_axes = self.joy_stick_data.axes
                joy_timestamp = str(self.joy_stick_data.header.stamp.sec) + "|" + str(self.joy_stick_data.header.stamp.nanosec)
                
                joy_dict = {"joy_buttons": list(joy_buttons), "joy_axes": list(joy_axes), "joy_timestamp": joy_timestamp}
                with open(self.dataset_subdir + "_joy.json", 'a') as f:
                    json.dump(joy_dict, f)
                    f.write("\n")


                self.img_count += 1
            self.condition.release()

    def update(self, im, im_timestamp, 
               vel_lin_x, vel_lin_y, vel_lin_z, vel_ang_x, vel_ang_y, vel_ang_z, joy_stick_data
               ):
        self.condition.acquire()
        self.im = im
        self.im_timestamp = im_timestamp
        self.vel_lin_x = vel_lin_x
        # adding other lin and ang values here
        self.vel_lin_y = vel_lin_y
        self.vel_lin_z = vel_lin_z
        self.vel_ang_x = vel_ang_x
        self.vel_ang_y = vel_ang_y
        self.vel_ang_z = vel_ang_z
        self.joy_stick_data = joy_stick_data
        self.condition.notify()
        self.condition.release()

    def stop(self):
        self.done = True
        self.update(0, 0, 0, 0, 0, 0)
        self.join()

class ROSbotDatasetWriterNode(Node):
    def __init__(self):
        super().__init__('rosbot_dataset_writer_node')
        self.image = None
        self.speed_cmd = 0.0
        self.turn_cmd = 0.0
        self.collecting = False
        self.vel_state = Twist()
        self.image_height = None
        self.image_width = None
        self.destination_folder = "/home/husarion/data"
        self.joy_stick_data = Joy()

        self.create_subscription(Image, '/image_raw', self.img_callback, 1)
        self.create_subscription(Twist, '/cmd_vel', self.velocity_callback, 1)
        self.create_subscription(Joy, '/joy', self.joy_call_back, 1)

        self.imwrite_thread = None
        # self.bridge = CvBridge()
        self.rate = self.create_rate(10)
        self.create_timer(0.1, self.main_loop)

    def img_callback(self, msg):
        #print("img_callbacking")
        #print(self.image)
        self.image = msg
        if (self.image_width is None or self.image_height is None) and self.image:
            print(f"setting image width to {msg.width} height to {msg.height}")
            self.image_width = msg.width
            self.image_height = msg.height
    def prepare_collecting(self):
        # create file based on current time
        self.dataset_subdir = "{}/{}".format(self.destination_folder, datetime.now().strftime("%Y%m%d_%H%M%S"))
        print(f"creating subdir at {self.dataset_subdir}")
        if not os.path.exists(self.dataset_subdir):
            og_mask = os.umask(0)
            os.makedirs(self.dataset_subdir, mode=0o777)
            os.umask(og_mask)
        if not self.imwrite_thread:
        	self.imwrite_thread = ImWriteThread(self.dataset_subdir)


    def joy_call_back(self, msg):
        self.joy_stick_data = msg
        if msg.buttons[4] == 1 and not self.collecting:
            self.collecting = True
            print("collecting", self.collecting)
            if self.collecting:
                self.prepare_collecting()
                self.imwrite_thread.timeout = 0.2
                time.sleep(0.1)
        elif msg.buttons[0] == 1 and self.collecting:
            self.collecting = False
            print("stop collecting")
            self.imwrite_thread.timeout = None
            self.imwrite_thread = None


    def velocity_callback(self, msg):
        self.vel_state = msg


    def cmd_vel_callback(self, msg):
        self.speed_cmd = msg.linear.x
        self.turn_cmd = msg.angular.z

    def main_loop(self):
        print("main_looping")
        #print("current data", self.vel_state)
        #print(self.vel_state.linear.x)
        #print(self.vel_state.linear.y)
        #print(self.vel_state)
        try:
        
            if self.image and self.collecting:
                print("start collecting")

                image_reshaped = np.frombuffer(self.image.data, dtype=np.uint8).reshape(self.image.height, self.image.width, -1)        
                # image is in YUV422 format, convert to RGB
                #image_reshaped = cv2.cvtColor(image_reshaped, cv2.COLOR_YUV2RGB_YUYV)
                
                print("image_reshaped", image_reshaped.shape)  
                
                print(image_reshaped.shape)
                self.imwrite_thread.update(
                    image_reshaped,
                    f"{self.image.header.stamp.sec}:{self.image.header.stamp.nanosec}", 
                    # self.speed_cmd, self.turn_cmd, self.batt_state, 
                    self.vel_state.linear.x,
                    # adding other vel values here
                    self.vel_state.linear.y,
                    self.vel_state.linear.z,
                    self.vel_state.angular.x,
                    self.vel_state.angular.y,
                    self.vel_state.angular.z, 
                    self.joy_stick_data)
			
                
               
                        
                if self.imwrite_thread.img_count % 10 == 0:
                	
                    self.get_logger().info("Dataset size=" + str(self.imwrite_thread.img_count))
                
        except Exception as e:
            print(e)
            self.get_logger().error(str(e))

def main(args=None):
    rclpy.init(args=args)
    node = ROSbotDatasetWriterNode()
    print("finished initialization")
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass

    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()

