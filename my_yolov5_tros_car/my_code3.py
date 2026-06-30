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
        self.subscription_image = self.create_subscription(Image, 'image', self.image_callback, 10)
        self.subscription_image  # prevent unused variable warning
        self.cv_image = None
          
        self.subscription = self.create_subscription(PerceptionTargets,'hobot_dnn_detection',self.listener_callback,10)
        self.subscription  # prevent unused variable warning
        self.count = 0
        self.fps = 0
        
        self.detections = []

    def image_callback(self, msg):
        # self.get_logger().info("Receiving video frame")
        if msg.encoding == 'jpeg':
            np_arr = np.frombuffer(msg.data,np.uint8)
            self.cv_image = cv2.imdecode(np_arr,cv2.IMREAD_COLOR)
        self.draw_detections()

    def listener_callback(self, msg):
        self.detections = msg.targets
        self.fps = msg.fps
        print("\n \033[31m---\033[0m This Frame: FPS = %d  \033[31m---\033[0m"%msg.fps)
        for num, target in enumerate(msg.targets):
            print("Traget \033[0;32;40m%d\033[0m: "%num, end="")
            print("Type: %s, x_offset=%d, y_offset=%d, height=%d, width=%d, conf=%.2f"%(target.rois[0].type,
            target.rois[0].rect.x_offset,
            target.rois[0].rect.y_offset,
            target.rois[0].rect.height,
            target.rois[0].rect.width,
            target.rois[0].confidence))

    def draw_detections(self):
        #if self.cv_image is not None and self.detections:
        # 在图像上绘制目标框
        typr_colors = {"person": (0, 255, 0),"rider": (0, 0, 255), "car": (255, 0, 0), "bus": (255, 255, 0),  "truck": (0, 255, 255), "bike": (255, 0, 255),  "motor": (0, 128, 128), "tl_green": (128, 0, 128),  "tl_red": (128, 128, 0), "tl_yellow": (0, 128, 0),   "tl_none": (128, 0, 0), "t_sign": (0, 0, 128), "train": (128, 128, 128) }
        for target in self.detections:
            xmin = target.rois[0].rect.x_offset
            ymin = target.rois[0].rect.y_offset
            xmax = xmin + target.rois[0].rect.width
            ymax = ymin + target.rois[0].rect.height
            target_type = target.rois[0].type
            confidence = target.rois[0].confidence
            color = typr_colors.get(target_type,(255,255,255))
            cv2.rectangle(self.cv_image, (xmin, ymin), (xmax, ymax),color, 2)
            cv2.putText(self.cv_image,f'{target_type}:{confidence:.2f}',(xmin,ymin-5),cv2.FONT_HERSHEY_SIMPLEX,0.5,color,2)
            if target_type == 'car':
                car_width_pixel = xmax-xmin
                focal_length = 500
                real_car_width = 1
                distance = (focal_length * real_car_width) / car_width_pixel
                distance_str = f'Dis:{distance:.2f} m'
                cv2.putText(self.cv_image,distance_str,(xmin,ymin-25),cv2.FONT_HERSHEY_SIMPLEX,0.5,color,2)
        cv2.putText(self.cv_image,f'FPS:{self.fps}',(10,20),cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,255),4)
        # 显示图像
        cv2.imshow("image", self.cv_image)
        cv2.waitKey(1)

def main(args=None):
    rclpy.init(args=args)
    image_subscriber = ImageSubscriber()
    rclpy.spin(image_subscriber)
    image_subscriber.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()

