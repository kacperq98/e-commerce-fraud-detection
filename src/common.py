import os 
import matplotlib.pyplot as plt
import logging

logger = logging.getLogger(__name__)

def check_if_path_exists(folder_path = None):
    try:
        os.makedirs(folder_path, exist_ok=True) 
    except Exception as e:
        logger.error(f"Can not create the folder path: {folder_path}. Error: {e}")


def save_plot(folder_path = None, file_name = None):
    try: 
        full_path = f'{folder_path}/{file_name}'

        check_if_path_exists(folder_path)

        plt.savefig(full_path, dpi=300, bbox_inches='tight')

        logger.info(f"Plot saved in: {full_path}")

    except Exception as e:
        logger.error(f"Can not save the plot. Error: {e}")

def make_logs_prettier(message: str, separator_lines: int = 1):
    for _ in range(separator_lines):
        print("--------------------------------------------------------------")
    
    logger.info(message)