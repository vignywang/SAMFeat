# 
# Created by Yuyang on 19-3-31
#

import logging
import time


def get_logger(log_root):
    # create a logger
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    # create a formatter
    formatter = logging.Formatter(
        fmt='%(asctime)s [%(levelname)s]: %(message)s', datefmt='%Y-%m-%d %H:%M:%S'
    )
    # writing
    log_dir = log_root + "/"
    c_t = time.strftime('%Y-%m-%d %H%M%S', time.localtime(time.time()))
    # if not os.path.exists(log_dir):
    #     os.mkdir(log_dir)
    log_name = log_dir + c_t + '.log'
    handler_file = logging.FileHandler(log_name, mode='w')
    handler_file.setFormatter(formatter)

    # showing
    handler = logging.StreamHandler()
    handler.setFormatter(formatter)

    logger.addHandler(handler_file)
    logger.addHandler(handler)
    return logger
