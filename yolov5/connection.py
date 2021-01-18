import os
import pathlib
import pickle
import shutil
import threading
import time
from queue import Queue
from socket import socket

import PIL
from PIL import Image
from scipy import sparse


def update_current_folder():
    folder: str = str(int(time.time()))
    pathlib.Path(f"{dir_path}\\{folder}\\").mkdir(parents=True, exist_ok=True)
    return folder


def execute_detect(current_folder):
    os.system(
        f'python detect.py '
        f'--weights runs/train/bestrun/weights/last.pt '
        f'--send '
        f'--img 416 '
        f'--conf 0.4 '
        f'--source "{dir_path}\\{current_folder}" '
        f'--project "{dir_path}\\{current_folder}" '
        f'--name "run_result"'
    )


def read_then_parse_then_save_if_enough_detect_send_and_delete(print_logs=False):
    def pprint(string):
        if print_logs:
            print(string)

    index = 0
    current_folder = update_current_folder()
    while True:
        sparse_bytes = queue.get(block=True, timeout=None)  # Blocking
        pprint(f"{len(sparse_bytes)}: {sparse_bytes}")
        if sparse_bytes is None:
            return

        # Detect, send and remove folder
        if index >= max_index:
            execute_detect(current_folder)
            shutil.rmtree(f"{dir_path}\\{current_folder}\\")  # Remove old folder
            current_folder = update_current_folder()
            index = 0

        # Save image to OI
        try:
            image_array = sparse.csr_matrix.toarray(pickle.loads(sparse_bytes))
        except pickle.UnpicklingError as e:
            pprint(f'[{e.__class__.__name__}] Occurred. \n\tLength: {len(sparse_bytes)}\n\tMessage:{sparse_bytes}')
            continue
        new_image: Image = PIL.Image.fromarray(image_array)
        new_image.save(f"{dir_path}\\{current_folder}\\{index}.jpg")

        index += 1


def receive_all(sock) -> bytes:
    """
    Receive all bytes from a socket.

    Author: JadedTuna -> https://stackoverflow.com/a/17697651/7230293
    """
    buff_size = 4096  # 4 KiB
    data = b''
    while True:
        part = sock.recv(buff_size)
        data += part
        if len(part) < buff_size:
            break  # either 0 or end of data
    return data


host = ""
port = 7879
timeout = 1

server_socket = socket()
server_socket.bind((host, port))
server_socket.listen()

print("=============================")
print("Server started.")
print(f"Listening on: {host}:{port}")
print("=============================")

relative_temp_images_folder = "\\temp_images\\"  # Relative path
dir_path = os.path.dirname(os.path.realpath(__file__)) + relative_temp_images_folder  # Absolute path

max_index: int = 80
queue = Queue()

parse_thread = threading.Thread(target=read_then_parse_then_save_if_enough_detect_send_and_delete, args=(True,))
parse_thread.start()

while True:
    try:
        client, address = server_socket.accept()

        received_bytes = receive_all(client)
        if len(received_bytes) > 0:
            queue.put(received_bytes)
    except IOError as e:
        print("[IOError] Occurred. Retrying in 5 seconds...")
        time.sleep(5)
