import rclpy
from rclpy.node import Node
from ai_msgs.msg import PerceptionTargets
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from PyQt5.QtCore import pyqtSignal
import cv2
import numpy as np

class ImageSubscriber(Node):
   
    def __init__(self):
        super().__init__('image_subscriber')
        self.subscription_image = self.create_subscription(Image, 'image', self.image_callback, 10)
        self.subscription_image  # prevent unused variable warning
        self.cv_image=None

          
        self.subscription = self.create_subscription(PerceptionTargets,'hobot_dnn_detection',self.listener_callback,10)
        self.subscription  # prevent unused variable warning
        self.count = 0
        self.fps = 0
        
        self.detections = []
        self.target_type = None
        self.confidence = None

    def image_callback(self, msg):
        if msg.encoding == 'jpeg':
            np_arr = np.frombuffer(msg.data,np.uint8)
            self.cv_image = cv2.imdecode(np_arr,cv2.IMREAD_COLOR)
            #print((self.cv_image.shape))
        self.draw_detections()
        #return self.cv_image
        
       
    def listener_callback(self, msg):
        self.detections = msg.targets
        self.fps = msg.fps

    def calculate_distance(self, target_type, pixel_width):
        # 定义不同类型物体的实际宽度
        object_widths = {
            'person': 0.5,  # 人的宽度为0.5m
            'rider': 0.5,   # 骑车人的宽度为0.5m
            'bike': 0.5,    # 自行车的宽度为0.5m
            'motor': 0.5,   # 摩托车的宽度为0.5m
            'car': 1.5,     # 轿车的宽度为1.5m
            'bus': 2.0,     # 巴士的宽度为2.0m
            'truck': 2.0    # 卡车的宽度为2.0m
        }

        # 根据相似三角形原理计算距离
        focal_length = 500  # 焦距，单位为像素
        real_object_width = object_widths.get(target_type, 1.0)  # 获取物体对应的实际宽度，默认为1.0m
        distance = (focal_length * real_object_width) / pixel_width
        return distance

    def draw_detections(self):
        typr_colors = {"person": (0, 255, 0), "rider": (0, 0, 255), "car": (255, 0, 0), "bus": (255, 255, 0),
                       "truck": (0, 255, 255), "bike": (255, 0, 255), "motor": (0, 128, 128)}
        
        for target in self.detections:
            xmin = target.rois[0].rect.x_offset
            ymin = target.rois[0].rect.y_offset
            xmax = xmin + target.rois[0].rect.width
            ymax = ymin + target.rois[0].rect.height
            self.target_type = target.rois[0].type
            self.confidence = target.rois[0].confidence
            color = typr_colors.get(self.target_type, (255, 255, 255))
            cv2.rectangle(self.cv_image, (xmin, ymin), (xmax, ymax), color, 2)
            cv2.putText(self.cv_image, f'{self.target_type}: {self.confidence:.2f}', (xmin, ymin - 5), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, color, 2)
            
            # 如果目标类型为需要测距的物体类型，则进行距离计算并显示
            if self.target_type in ['person', 'rider', 'bike', 'motor', 'car', 'bus', 'truck']:
                pixel_width = target.rois[0].rect.width
                distance = self.calculate_distance(self.target_type, pixel_width)
                cv2.putText(self.cv_image, f'Distance: {distance:.2f}m', (xmin, ymax + 20), cv2.FONT_HERSHEY_SIMPLEX,
                            0.5, color, 2)
        cv2.putText(self.cv_image, f'FPS: {self.fps}', (30, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (128, 128, 255), 5)
        #cv2.imshow("image", self.cv_image)
        #cv2.waitKey(1)
        
    def get_image(self):
        current_image = self.cv_image
        self.cv_image = None
        return current_image  
        
    def get_msg(self):
        target_type = self.target_type  
        confidence = self.confidence  
        fps = self.fps  
        current_msg = [fps,target_type,confidence]       
        return current_msg
          

def main(args=None):
    rclpy.init(args=args)
    image_subscriber = ImageSubscriber()
    rclpy.spin(image_subscriber)
    image_subscriber.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
