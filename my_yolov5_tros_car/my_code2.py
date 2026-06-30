import rclpy
from rclpy.node import Node
from ai_msgs.msg import PerceptionTargets
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import numpy as np

class ImageSubscriber(Node):
    def __init__(self):
        super().__init__('image_subscriber')
        self.subscription = self.create_subscription(Image,'image',self.listener_callback,10)
        self.subscription  # prevent unused variable warning
        self.cv_bridge = CvBridge()

    def listener_callback(self, msg):
        self.get_logger().info("Receiving video frame")
        # cv_image = self.cv_bridge.imgmsg_to_cv2(msg,'passthrough')
        if msg.encoding == 'jpeg':
            np_arr = np.frombuffer(msg.data,np.uint8)
            cv_image = cv2.imdecode(np_arr,cv2.IMREAD_COLOR)
        cv2.imshow("image",cv_image)
        cv2.waitKey(1)

def main(args=None):
    rclpy.init(args=args)
    image_subscriber = ImageSubscriber()
    rclpy.spin(image_subscriber)
    image_subscriber.destroy_node()
    rclpy.shutdown()

# 给main函数一个入口，省得colcon build编译
if __name__ == '__main__':
    main()

