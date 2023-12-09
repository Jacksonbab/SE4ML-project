#!/usr/bin/env python
from datetime import datetime
import os
import random
import string
import time
import csv
import numpy as np
import cv2
from cv_bridge import CvBridge
import threading

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

header = ['linear_x', 'linear_y', 'linear_z', 'angular_x','angular_y', 'angular_z']
with open('/home/husarion/data/drive_data.csv', 'w', encoding='UTF8', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(header)    

class ImWriteThread(threading.Thread):
    def __init__(self, dataset_subdir):
        super(ImWriteThread, self).__init__()
        self.im = None
        self.im_timestamp = None
        self.img_filename = ""
        self.speed_cmd = None
        self.turn_cmd = None
        self.batt_state = None
        self.vel_lin_x = None
        # adding other lin and ang values here
        self.vel_lin_y = None
        self.vel_lin_z = None
        self.vel_ang_x = None
        self.vel_ang_y = None
        self.vel_ang_z = None
        self.lidar_angle_min = None
        self.lidar_angle_max = None
        self.lidar_angle_increment = None
        self.lidar_range_min = None
        self.lidar_range_max = None
        self.lidar_ranges = None
        self.lidar_intensities = None
        self.range_fl = None
        self.range_fr = None
        self.range_rl = None
        self.range_rr = None
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
            #self.img_filename = "astra-{:05d}.jpg".format(self.img_count)
            #cv2.imwrite("{}/{}".format(self.dataset_subdir, self.img_filename), self.im)
            print("logging")
            with open(self.dataset_subdir + "/data.csv", 'a') as f:
                f.write("{},{},{},{},{},{},{},{}\n".format(self.img_filename, self.im_timestamp,  self.vel_lin_x, 
                # adding other lin and ang values here
                self.vel_lin_y,
                self.vel_lin_z,
                self.vel_ang_x,
                self.vel_ang_y,         
                self.vel_ang_z))
                self.img_count += 1
            self.condition.release()

    def update(self, im_timestamp, speed_cmd, turn_cmd, batt_state, 
               vel_lin_x, vel_lin_y, vel_lin_z, vel_ang_x, vel_ang_y, vel_ang_z):
        self.condition.acquire()
        self.im_timestamp = im_timestamp
        self.speed_cmd = speed_cmd
        self.turn_cmd = turn_cmd
        self.batt_state = batt_state
        self.vel_lin_x = vel_lin_x
        # adding other lin and ang values here
        self.vel_lin_y = vel_lin_y
        self.vel_lin_z = vel_lin_z
        self.vel_ang_x = vel_ang_x
        self.vel_ang_y = vel_ang_y
        self.vel_ang_z = vel_ang_z
        
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
        self.batt_state = None
        self.vel_state = None
        self.lidar_state = None
        self.range_fl = None
        self.range_rl = None
        self.range_fr = None
        self.range_rr = None

        #self.create_subscription(Image, '/zed_camera/rgb/left_mage', self.img_callback, 1)
        self.create_subscription(BatteryState, '/battery', self.battery_callback, 1)
        self.create_subscription(Twist, '/cmd_vel', self.velocity_callback, 1)
        #self.create_subscription(LaserScan, '/scan', self.lidar_callback, 1)
        #self.create_subscription(Range, '/range/fl', self.range_fl_callback, 1)
        #self.create_subscription(Range, '/range/fr', self.range_fr_callback, 1)
        #self.create_subscription(Range, '/range/rl', self.range_rl_callback, 1)
        #self.create_subscription(Range, '/range/rr', self.range_rr_callback, 1)
        self.create_subscription(Twist, '/cmd_vel', self.cmd_vel_callback, 1)

        destination_folder="/home/husarion/data"
        self.imwrite_thread = ImWriteThread(destination_folder)
        self.bridge = CvBridge()
        self.rate = self.create_rate(1)
        self.create_timer(1, self.main_loop)

    def img_callback(self, msg):
        print("img_callbacking")
        self.image = msg

    def battery_callback(self, msg):
        self.batt_state = msg.voltage

    def velocity_callback(self, msg):
        self.vel_state = msg

    def lidar_callback(self, msg):
        self.lidar_state = msg

    def range_fl_callback(self, msg):
        self.range_fl = msg.range

    def range_fr_callback(self, msg):
        self.range_fr = msg.range

    def range_rl_callback(self, msg):
        self.range_rl = msg.range

    def range_rr_callback(self, msg):
        self.range_rr = msg.range

    def cmd_vel_callback(self, msg):
        self.speed_cmd = msg.linear.x
        self.turn_cmd = msg.angular.z

    def main_loop(self):
        print("start collection")
        #print("current data", self.vel_state)
       # print(self.vel_state.linear.x)
        #print(self.vel_state.linear.y)
        print(self.vel_state)
        with open('/home/husarion/data/drive_data.csv', 'a', encoding='UTF8', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([str(self.vel_state)])
        try:
            collecting = True
            print("start saving")
            if collecting:
                print("start saving")
                #bridge_img = self.bridge.imgmsg_to_cv2(self.image, desired_encoding='passthrough')[...,::-1]                  
                
                self.imwrite_thread.update(
                    #bridge_img, 
                    f"{datetime.now()}", 
                    self.speed_cmd, self.turn_cmd, self.batt_state, 
                    self.vel_state.linear.x,
                    # adding other vel values here
                    self.vel_state.linear.y,
                    self.vel_state.linear.z,
                    self.vel_state.angular.x,
                    self.vel_state.angular.y,
                    self.vel_state.angular.z, )
			
                #data = [self.vel_state.linear.x, self.vel_state.linear.y, self.vel_state.linear.z,   self.vel_state.angular.x, self.vel_state.angular.y, self.vel_state.angular.z] 
               
                        
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

