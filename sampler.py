import dearpygui.dearpygui as dpg
import os
from PIL import Image
import numpy as np
import random
import time
import threading
import math

dpg.create_context()
dpg.create_viewport()
dpg.setup_dearpygui()

screen_width = 1920
screen_height = 1080

img_width = 800
img_height = 400

img_x = 200
img_y = 200

start_time = time.time()

# get images address
dataset_path = "D:\\school\\videa\\seaPics"
filenames = []
video_to_frame_indices_map = {}
frame_idx_to_frame_path_map = {}

idx = 0
for dirname in sorted(os.listdir(dataset_path)):
  dirpath = os.path.join(dataset_path, dirname)
  video_indices = []
  for fn in sorted(os.listdir(dirpath)):
    filename = os.path.join(dirpath, fn)
    filenames.append(filename)
    video_indices.append(idx)
    frame_idx_to_frame_path_map[idx] = filename
    idx += 1
  video_to_frame_indices_map[dirpath] = video_indices


def get_random_dpg_image(return_filename = False):
  roll = random.randint(0, len(filenames) - 1)
  image = Image.open(filenames[roll]).resize((img_width, img_height))
  image.putalpha(255)
  dpg_image = np.frombuffer(image.tobytes(), dtype=np.uint8) / 255.0

  if return_filename:
    return (dpg_image, filenames[roll])
  return dpg_image

def next_img_shortcut():
  global start_time

  img, filename = get_random_dpg_image(True)

  dpg.set_value("tex", img)
  diff = round(time.time() - start_time, 2)
  print(f"elapsed time: {diff} s")
  print(f"showing {filename}")
  start_time = time.time()

def run_time_updater():
  diff = round(time.time() - start_time, 2)
  dpg.set_value("time", f"{diff} s")
  threading.Timer(0.1, run_time_updater).start()

def next_img_static_shortcut():
  global start_time

  dpg.delete_item("img")
  dpg.delete_item("tex")
  
  img, filename = get_random_dpg_image(True)
  with dpg.texture_registry():
    dpg.add_static_texture(width=img_width, height=img_height, default_value=img, tag="tex")
  dpg.add_image("tex", tag="img", pos=[img_x, img_y], parent=window)

  diff = round(time.time() - start_time, 2)
  print(f"elapsed time: {diff} s")
  print(f"showing {filename}")
  start_time = time.time()

with dpg.texture_registry():
  dpg.add_static_texture(width=img_width, height=img_height, default_value=get_random_dpg_image(), tag="tex")

with dpg.handler_registry():
  dpg.add_key_press_handler(key=dpg.mvKey_Return, callback=next_img_static_shortcut)

with dpg.window(label="Image Window", width=screen_width, height=screen_height, no_collapse=True, no_resize=True, no_close=True, no_move=True, no_title_bar=True, pos=[0, 0]) as window:
  dpg.add_text("0 s", tag="time")  
  dpg.add_image("tex", tag="img", pos=[img_x, img_y])

run_time_updater()

dpg.show_viewport()
dpg.toggle_viewport_fullscreen()
dpg.start_dearpygui()
dpg.destroy_context()
