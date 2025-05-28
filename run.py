import numpy as np
import torch
import argparse
import os

from src import config
from src.slam import SLAM
from src.utils.datasets import get_dataset
from time import gmtime, strftime
from colorama import Fore,Style

import random

# **** qingshufan modified code start ****
import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import re
# **** qingshufan modified code end ****

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

# **** qingshufan modified code start ****
class ImageSaver:
    def __init__(self, save_folder):
        self.save_folder = save_folder
        self.bridge = CvBridge()
        self.frame_counter = 0
        self.start = False
        self.last_msg_time = rospy.Time.now()

        os.makedirs(save_folder, exist_ok=True)
        
        self.sub = rospy.Subscriber(
            '/camera/rgb/image_color', 
            Image, 
            self.image_callback, 
            queue_size=10
        )
        
        rospy.loginfo(f"Images will be saved to: {save_folder}")
        rospy.loginfo(f"Starting frame number: {self.frame_counter}")

    def image_callback(self, msg):
        self.last_msg_time = rospy.Time.now()
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            
            filename = os.path.join(
                self.save_folder, 
                f"frame{self.frame_counter:06d}.png"
            )
            
            cv2.imwrite(filename, cv_image)
            rospy.loginfo(f"Image saved: {filename}")
            self.start = True
            
            self.frame_counter += 1
        except Exception as e:
            rospy.logerr(f"Failed to save image: {e}")

    def check_timeout(self):
        if not self.start:
            return False
        if rospy.Time.now() - self.last_msg_time > rospy.Duration(1):
            rospy.loginfo("Received completed!")
            rospy.signal_shutdown("Received completed!") 
            return True
            
        return False

 # **** qingshufan modified code end ****
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('config', type=str, help='Path to config file.')   
    # **** qingshufan modified code start ****
    parser.add_argument('--ros', action='store_true', help='Enable ROS mode')
    # **** qingshufan modified code end ****
    args = parser.parse_args()

    torch.multiprocessing.set_start_method('spawn')

    cfg = config.load_config(args.config)

    # **** qingshufan modified code start ****
    #ROS
    if args.ros:
        try:
            rospy.init_node('semantic_wildgs_slam', anonymous=True)
            save_folder = os.path.join(
                cfg['data']['input_folder'].replace("ROOT_FOLDER_PLACEHOLDER", cfg['data']['root_folder']), 
                'rgb'
            )
            saver = ImageSaver(save_folder)
            rospy.Timer(rospy.Duration(1), lambda event: saver.check_timeout())
            rospy.spin()
        except Exception as e:
            print(f"ROS node failed: {str(e)}")
    else:
        print("ROS mode disabled. Running in standalone mode.")
    # **** qingshufan modified code end ****
    
    setup_seed(cfg['setup_seed'])
    if cfg['fast_mode']:
        # Force the final refine iterations to be 3000 if in fast mode
        cfg['mapping']['final_refine_iters'] = 3000
    
    output_dir = cfg['data']['output']
    output_dir = output_dir+f"/{cfg['scene']}"

    start_time = strftime("%Y-%m-%d %H:%M:%S", gmtime())
    start_info = "-"*30+Fore.LIGHTRED_EX+\
                 f"\nStart WildGS-SLAM at {start_time},\n"+Style.RESET_ALL+ \
                 f"   scene: {cfg['dataset']}-{cfg['scene']},\n" \
                 f"   output: {output_dir}\n"+ \
                 "-"*30
    print(start_info)
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    config.save_config(cfg, f'{output_dir}/cfg.yaml')

    dataset = get_dataset(cfg)

    slam = SLAM(cfg,dataset)
    slam.run()

    end_time = strftime("%Y-%m-%d %H:%M:%S", gmtime())
    print("-"*30+Fore.LIGHTRED_EX+f"\nWildGS-SLAM finishes!\n"+Style.RESET_ALL+f"{end_time}\n"+"-"*30)

