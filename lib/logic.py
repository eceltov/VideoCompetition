import os
from PIL import Image
import pickle
import time
from lib.model import Model
import numpy as np
import array
import dearpygui.dearpygui as dpg

class Logic:
  def __init__(self, shown, button_width, button_height, corner_width, corner_height) -> None:
    self.dataset_path = "C:\\school\\MVK"
    self.shown = shown
    self.button_width = button_width
    self.button_height = button_height
    self.corner_width = corner_width
    self.corner_height = corner_height
    self.model = Model()

    # get images address
    self.filenames = []
    self.annotated_filenames = []
    self.video_to_frame_indices_map = {}
    self.frame_idx_to_frame_path_map = {}

    idx = 0
    for dirname in sorted(os.listdir(self.dataset_path)):
      dirpath = os.path.join(self.dataset_path, dirname)
      video_indices = []
      for fn in sorted(os.listdir(dirpath)):
        filename = os.path.join(dirpath, fn)
        self.filenames.append(filename)
        video_indices.append(idx)
        self.frame_idx_to_frame_path_map[idx] = filename
        idx += 1
      self.video_to_frame_indices_map[dirpath] = video_indices

    for filename in self.filenames:
      annotated_filename = filename.replace("MVK", "MVKout")
      self.annotated_filenames.append(annotated_filename)


    # contains top_result lists of previous actions
    self.history = []
    self.history.append(range(shown))
    self.history_idx = 0

    # frames that did not fit on the screen
    self.not_shown_video_frames = []

    self.images_buttons = []
    self.selected_images = []

  def savePreprocessImages(self):
    # preprocess images to speed up the search engine
    image_cache = []
    for filename in self.filenames:
      image = Image.open(filename).resize((self.button_width, self.button_height))
      image.putalpha(255)
      dpg_image = np.frombuffer(image.tobytes(), dtype=np.uint8) / 255.0
      image_cache.append(array.array('f', dpg_image))

    with open('image_cache_dpg_raw.pickle', 'wb') as handle:
      pickle.dump(image_cache, handle, protocol=pickle.HIGHEST_PROTOCOL)

  def loadPreprocessedImages(self):
    with open('image_cache_dpg_raw.pickle', 'rb') as handle:
      return pickle.load(handle)
    
  def get_dpg_image(self, idx, annotate=False):
    filename = self.filenames[idx]
    if annotate:
      filename = self.annotated_filenames[idx]
    
    image = Image.open(filename).resize((self.button_width, self.button_height))
    image.putalpha(255)
    dpg_image = np.frombuffer(image.tobytes(), dtype=np.uint8) / 255.0
    return dpg_image
  
  def get_dpg_image_corners(self, idx):
    corners = self.model.get_image_corners(self.filenames[idx])
    resized = [img.resize((self.corner_width, self.corner_height)) for img in corners]
    for corner in resized:
      corner.putalpha(255)
    
    dpg_corners = [np.frombuffer(corner.tobytes(), dtype=np.uint8) / 255.0 for corner in resized]
    return dpg_corners
  
  def get_resized_dpg_image(self, idx, width, height, annotate=False):
    filename = self.filenames[idx]
    if annotate:
      filename = self.annotated_filenames[idx]

    image = Image.open(filename).resize((width, height))
    image.putalpha(255)
    dpg_image = np.frombuffer(image.tobytes(), dtype=np.uint8) / 255.0
    return dpg_image
  
  def append_history(self, indices):
    # remove 'future' searches if a new forward branch is made
    self.history = self.history[:self.history_idx + 1]
    self.history_idx += 1

    # append image indices and create new images
    self.history.append(indices)

  def blacklist_shown_frames(self):
    frames = self.history[self.history_idx]
    self.model.set_scores_to_zero(frames)
    self.append_history(self.model.get_top_score_indices(self.shown)[:self.shown].tolist())

  def blacklist_shown_videos(self):
    frames = self.history[self.history_idx]
    for imgIdx in frames:
      img_path = self.frame_idx_to_frame_path_map[imgIdx]
      dir_path = os.path.dirname(img_path)
      video_frames = self.video_to_frame_indices_map[dir_path]
      self.model.set_scores_to_zero(video_frames)
    self.append_history(self.model.get_top_score_indices(self.shown)[:self.shown].tolist())
    