import time

def get_current_time():
    return time.strftime('%Y-%m-%d_%H-%M-%S', time.localtime())