# Copyright 2024 The Splat-SLAM Authors.
# Licensed under the Apache License, Version 2.0
# available at: https://github.com/google-research/Splat-SLAM/blob/main/LICENSE
 
from colorama import Fore, Style
import torch.multiprocessing as mp


class FontColor(object):
    MAPPER=Fore.CYAN
    TRACKER=Fore.BLUE
    INFO=Fore.YELLOW
    ERROR=Fore.RED
    PCL=Fore.GREEN
    EVAL=Fore.MAGENTA
    MESH="yellow"


def get_msg_prefix(color):
    if color == FontColor.MAPPER:
        msg_prefix = color + "[MAPPER] " + Style.RESET_ALL
    elif color ==  FontColor.TRACKER:
        msg_prefix = color + "[TRACKER] " + Style.RESET_ALL
    elif color ==  FontColor.INFO:
        msg_prefix = color + "[INFO] " + Style.RESET_ALL
    elif color ==  FontColor.ERROR:
        msg_prefix = color + "[ERROR] " + Style.RESET_ALL
    elif color ==  FontColor.PCL:
        msg_prefix = color + "[POINTCLOUD] " + Style.RESET_ALL
    elif color ==  FontColor.EVAL:
        msg_prefix = color + "[EVALUATION] " + Style.RESET_ALL
    elif color == FontColor.MESH:
        msg_prefix = FontColor.INFO + "[MESH] " + Style.RESET_ALL
    else:
        msg_prefix = Style.RESET_ALL
    return msg_prefix

class TrivialPrinter(object):
    def print(self,msg:str,color=None):
        msg_prefix = get_msg_prefix(color)
        msg = msg_prefix + msg + Style.RESET_ALL
        print(msg)        

class Printer(TrivialPrinter):
    def __init__(self, total_img_num):
        self.msg_lock = mp.Lock()
        self.msg_queue = mp.Queue()
        self.progress_counter = mp.Value('i', 0)
        self.total_img_num = mp.Value('i', 0)
        process = mp.Process(target=self.printer_process, args=(total_img_num,))
        process.start()
    def print(self,msg:str,color=None):
        msg_prefix = get_msg_prefix(color)
        msg = msg_prefix + msg + Style.RESET_ALL
        with self.msg_lock:
            self.msg_queue.put(msg)
    def update_pbar(self):
        with self.msg_lock:
            self.progress_counter.value += 1
            self.msg_queue.put(f"PROGRESS")
            
    # **** qingshufan modified code start ****
    def update_total(self,total_img_num):
        with self.msg_lock:
            self.total_img_num.value = total_img_num
    # **** qingshufan modified code end ****

    def pbar_ready(self):
        with self.msg_lock:
            self.msg_queue.put(f"READY")        

    def printer_process(self,total_img_num):
        from tqdm import tqdm

        # **** qingshufan modified code start ****

        self.total_img_num.value = total_img_num
        current_total = total_img_num 
        while True:
            message = self.msg_queue.get()
            if message == "READY":
                break
            else:
                print(message)
        
        with tqdm(total=current_total) as pbar:
            while self.progress_counter.value < current_total:
                message = self.msg_queue.get()
                if message == "DONE":
                    break
                elif message.startswith("PROGRESS"):
                    with self.msg_lock:
                        completed = self.progress_counter.value
                        if self.total_img_num.value != current_total:
                            pbar.total = self.total_img_num.value
                            current_total = self.total_img_num.value
                            pbar.refresh()
                    pbar.set_description(FontColor.TRACKER+f"[TRACKER] "+Style.RESET_ALL)
                    pbar.n = completed
                    pbar.refresh()
                else:
                    pbar.write(message)
        
        # **** qingshufan modified code end ****
        
        while True:
            
            message = self.msg_queue.get()
            if message == "DONE":
                break
            else:
                print(message)
            
    
    def terminate(self):
        self.msg_queue.put("DONE")


