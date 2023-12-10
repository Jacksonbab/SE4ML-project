import sys
sys.path.append("/home/husarion/ros2_tutorial_workspace/install/python_package_with_a_node/lib/python3.10/site-packages/python_package_with_a_node")

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist
from inference import *
import numpy as np



class RosbotML(Node):
    def __init__(self):
        super().__init__('rosbot_ml')
        self.model = load_model('/home/husarion/ros2_tutorial_workspace/src/python_package_with_a_node/python_package_with_a_node/model_cpt/exampler_cpt.pt')
        print("model loaded")
        self.image = None

        self.create_subscription(Image, '/image_raw', self.img_callback, 1)
        # create publisher for cmd_vel
        self.cmd_vel_pub = self.create_publisher(Twist, 'cmd_vel', 1)

        # create timer for self.run
        self.rate = self.create_rate(10)
        self.timer = self.create_timer(0.1, self.run)
       



    def img_callback(self, msg):
        #print(msg)
        self.image = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width, -1)
        print(self.image.shape)

    def run(self):
        if self.image is not None:
            print("inferencing")
            prediction = inference(self.model, self.image)
            print(prediction)
            # publish prediction
            msg = Twist()
            msg.linear.x = 0.2
            msg.angular.z = prediction
            self.cmd_vel_pub.publish(msg)
            #self.rate.sleep()

def main(args=None):
    rclpy.init(args=args)

    rosbot_ml = RosbotML()

    rclpy.spin(rosbot_ml)

    rosbot_ml.destroy_node()
    rclpy.shutdown()
                

if __name__ == '__main__':
    main()
