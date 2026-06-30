#!/bin/bash
cd /home/sunrise/Desktop/my_yolov5_tros_car/

gnome-terminal -- bash -c "source /opt/tros/setup.bash;export CAM_TYPE=usb;ros2 launch dnn_node_example dnn_node_example.launch.py dnn_example_config_file:=yolov5workconfig.json #dnn_example_image_width:=640 dnn_example_image_height:=480"

gnome-terminal -- bash -c "source /opt/tros/setup.bash; python3 test.py"


