import dearpygui.dearpygui as dpg
import os
from PIL import Image
import numpy as np
import pickle
import time
import array
from lib.logic import Logic

def get_texture_tag(row, col):
   return f"tex_{row}_{col}"

def get_img_tag(row, col):
   return f"img_{row}_{col}"

def get_border_tag(row, col):
   return f"rect_{row}_{col}"


dpg.create_context()
dpg.create_viewport()
dpg.setup_dearpygui()

shortcuts_disabled = False

screen_width = 1920
screen_height = 1080

tools_width = 200
images_width = screen_width - tools_width

button_cols = 6
button_rows = 8
shown = button_cols * button_rows
button_width = images_width // button_cols - 5
button_height = screen_height // button_rows - 5

corner_width = tools_width // 2
corner_height = corner_width // 2

yolo_width = tools_width
yolo_height = tools_width // 2
yolo_y_pos = screen_height - yolo_height - 50

print(yolo_width, yolo_height)

logic = Logic(shown, button_width, button_height, corner_width, corner_height)

previous_text = ""
previous_corner_texts = ["", "", "", ""]
previous_corner_weights = [1, 1, 1, 1]

def save_previous_inputs():
  global previous_text
  global previous_corner_texts
  global previous_corner_weights

  previous_text = dpg.get_value("search")
  for i in range(4):
    previous_corner_texts[i] = dpg.get_value(f"search_{i}")
  previous_corner_weights = logic.model.corner_weights  

def load_previous_inputs():
  dpg.set_value("search", previous_text)
  for i in range(4):
    dpg.set_value(f"search_{i}", previous_corner_texts[i])
  logic.model.corner_weights = previous_corner_weights  

def update_corner_weight_labels():
  for i in range(4):
    dpg.set_value(f"corner_weight_{i}", logic.model.corner_weights[i])

def increase_corner_weight_shortcut(idx):
  logic.model.corner_weights[idx] += 1
  update_corner_weight_labels()

def decrease_corner_weight_shortcut(idx):
  logic.model.corner_weights[idx] -= 1
  update_corner_weight_labels()

def update_alpha_label():
  dpg.set_value("alpha", logic.model.alpha)

def increase_alpha_shortcut():
  logic.model.alpha *= 2
  update_alpha_label()

def decrease_alpha_shortcut(idx):
  logic.model.alpha /= 2
  update_alpha_label()

def get_row_col_from_idx(idx):
  row = idx // button_cols
  col = idx % button_cols
  return (row, col)

def get_img_position(row, col):
  x = col * button_width + 3 * col + 5
  y = row * button_height + 3 * row + 5
  return (x, y)

def get_rect_position(row, col):
  x, y = get_img_position(row, col)
  x -= 8
  y -= 8
  return (x, y)  

def show_image(row, col):
  dpg.show_item(get_img_tag(row, col))

def hide_image(row, col):
  dpg.hide_item(get_img_tag(row, col))

def corners_contain_input():
  for i in range(4):
    text = dpg.get_value(f"search_{i}")
    if text != None and len(text) > 0:
      return True
  return False

previous_text_prompt = ""
previous_top_results = []
previous_prompt_repetition = 0
def search_clip_shortcut():
  global previous_text_prompt
  global previous_top_results
  global previous_prompt_repetition

  hide_input()

  if corners_contain_input():
    search_clip_corners_shortcut()
    return

  text = dpg.get_value("search")
  if text == previous_text_prompt:
    previous_prompt_repetition += 1
  else:
    previous_prompt_repetition = 0
  previous_text_prompt = text

  startTime = time.time()
  print(text, end=" ")

  if previous_prompt_repetition == 0:
    previous_top_results = logic.model.search_clip(text)

  logic.append_history(previous_top_results[(shown * previous_prompt_repetition):(shown * (previous_prompt_repetition + 1))].tolist())

  display_images()
  set_shortcuts_disabled(False)

  # hide full image if searching
  if viewing_full_image:
    view_full_image_shortcut()
  print(time.time() - startTime)

def search_clip_corners_shortcut():
  texts = [dpg.get_value(f"search_{i}") for i in range(4)]

  startTime = time.time()
  print(texts, end=" ")

  top_result = logic.model.search_clip_corners(texts)
  logic.append_history(top_result[:shown].tolist())

  display_images()
  set_shortcuts_disabled(False)

  # hide full image if searching
  if viewing_full_image:
    view_full_image_shortcut()
  print(time.time() - startTime)

def hide_images(start_idx):
  for i in range(start_idx, min(shown, displayed_images_count)):
    row, col = get_row_col_from_idx(i)
    hide_image(row, col)

def show_all_images():
  top_result = logic.history[logic.history_idx]
  for i in range(len(top_result)):
    row, col = get_row_col_from_idx(i)
    show_image(row, col)

def set_corner_image(corner_idx, img_idx):
  corner = logic.get_dpg_image_corners(img_idx)[corner_idx]
  dpg.set_value(f"corner_tex_{corner_idx}", corner)

def select_corner_shortcut(corner_idx):
  if len(logic.selected_images) <= 0:
    return
  
  img_idx = logic.selected_images[0]
  logic.model.selected_corner_images[corner_idx] = img_idx
  set_corner_image(corner_idx, img_idx)

def add_image_and_texture(img_idx, image):
  row, col = get_row_col_from_idx(img_idx)
  dpg.add_static_texture(width=button_width, height=button_height, default_value=image, tag=get_texture_tag(row, col), parent=registry)

  x, y = get_img_position(row, col)

  img_tag = get_img_tag(row, col)
  img_data = {
      "pos": [row, col],
      "rectId": rect_ids[img_idx],
      "imgIdx": row * button_cols + col
  }
  dpg.add_image(get_texture_tag(row, col), tag=img_tag, pos=[x, y], user_data=img_data, parent=window)
  dpg.bind_item_handler_registry(img_tag, "image click handler")


def remove_images_and_textures():
  global displayed_images_count

  for i in range(displayed_images_count):
    row, col = get_row_col_from_idx(i)
    dpg.delete_item(get_img_tag(row, col))
    dpg.delete_item(get_texture_tag(row, col))
  displayed_images_count = 0

displayed_images_count = button_cols * button_rows
def display_images():
  global displayed_images_count

  top_result = logic.history[logic.history_idx]
  images = [logic.get_dpg_image(top_result[i]) for i in range(len(top_result))]

  remove_images_and_textures()

  for i in range(len(top_result)):
    row, col = get_row_col_from_idx(i)
    add_image_and_texture(i, images[i])
    img_data = dpg.get_item_user_data(get_img_tag(row, col))
    img_data["imgIdx"] = top_result[i]
      
  displayed_images_count = len(top_result)
  hide_borders()
  update_non_zero_scores_text()

def display_images2():
  top_result = logic.history[logic.history_idx]
  images = [logic.get_dpg_image(top_result[i]) for i in range(len(top_result))]

  for i in range(len(top_result)):
    row, col = get_row_col_from_idx(i)
    show_image(row, col)
    image = images[i]
    dpg.set_value(get_texture_tag(row, col), image)
    img_data = dpg.get_item_user_data(get_img_tag(row, col))
    img_data["imgIdx"] = top_result[i]
      
  hide_images(len(top_result))
  hide_borders()
  update_non_zero_scores_text()

def go_back_shortcut():
  if shortcuts_disabled:
    return 

  if logic.history_idx <= 0:
    return
  
  logic.history_idx -= 1
  display_images()

def go_forward_shortcut():
  if shortcuts_disabled:
    return

  if logic.history_idx < len(logic.history) - 1:
    logic.history_idx += 1
    display_images()

def show_video_shortcut():
  if shortcuts_disabled:
    return

  if len(logic.selected_images) <= 0:
    if len(logic.not_shown_video_frames) > 0:
      logic.append_history(logic.not_shown_video_frames[:shown])
      logic.not_shown_video_frames = logic.not_shown_video_frames[shown:]
      display_images()
    return
  
  idx = logic.selected_images[0]
  path = logic.frame_idx_to_frame_path_map[idx]
  dirpath = os.path.dirname(path)
  frames = logic.video_to_frame_indices_map[dirpath]
  logic.append_history(frames[:shown])
  logic.not_shown_video_frames = frames[shown:]
  display_images()

def update_selected_images_text():
  if len(logic.selected_images) > 0:
    dpg.set_value("selected", str(logic.selected_images))
  else:
    dpg.set_value("selected", "")

def update_non_zero_scores_text():
  dpg.set_value("non zero scores", "Non Zero Scores: " + str(logic.model.non_zero_scores_count()))

def focus_input():
  dpg.focus_item("search")

def clear_input():
  dpg.set_value("search", "")
  for i in range(4):
    dpg.set_value(f"search_{i}", "")

def hide_input():
  dpg.hide_item("search")
  for i in range(4):
    dpg.hide_item(f"search_{i}")
    dpg.hide_item(f"corner_weight_{i}")
    dpg.hide_item(f"corner_weight_extra_{i}")

def set_shortcuts_disabled(disabled):
  global shortcuts_disabled

  shortcuts_disabled = disabled
  if disabled:
    dpg.set_value("shortcut status", "Shortcuts DISABLED")
  else:
    dpg.set_value("shortcut status", "Shortcuts ENABLED")

def focus_input_shortcut(clear_inputs):
  save_previous_inputs()
  if clear_inputs:
    clear_input()
    logic.model.corner_weights = [1, 1, 1, 1]
  focus_input()
  dpg.show_item("search")
  for i in range(4):
    dpg.show_item(f"search_{i}")
    dpg.show_item(f"corner_weight_{i}")
    dpg.show_item(f"corner_weight_extra_{i}")
  update_corner_weight_labels()
  set_shortcuts_disabled(True)

def hide_borders():
  for row in range(button_rows):
    for col in range(button_cols):
      dpg.hide_item(get_border_tag(row, col))
  logic.selected_images = []
  update_selected_images_text()

def img_click_callback(sender, app_data):
  img_data = dpg.get_item_user_data(app_data[1])
  
  rectId = img_data["rectId"]
  if dpg.is_item_shown(rectId):
    dpg.hide_item(rectId)
    logic.selected_images.remove(img_data["imgIdx"])
  else:
    hide_borders()
    dpg.show_item(rectId)
    logic.selected_images = [img_data["imgIdx"]]
  update_selected_images_text()

def yolo_click_callback(sender, app_data):
  position = dpg.get_mouse_pos(local=False)
  img_pos_x = position[0] - 4 # remove window margin
  img_pos_y = position[1] - yolo_y_pos


  print(img_pos_x, img_pos_y)

def score_search_shortcut():
  if shortcuts_disabled:
    return

  if len(logic.selected_images) <= 0:
    return

  likeID = logic.selected_images[0]
  logic.model.update_scores(logic.history[-1], likeID)
  logic.append_history(logic.model.get_top_score_indices(shown)[:shown].tolist())
  display_images()

last_corner_images = [1, 2, 3, 4]
corner_images_match_count = 0
last_corner_top_indices = []
def image_score_search_shortcut():
  global last_corner_images
  global corner_images_match_count
  global last_corner_top_indices

  if shortcuts_disabled:
    return
  
  matching_indices = True
  for i in range(4):
    if logic.model.selected_corner_images[i] != last_corner_images[i]:
      matching_indices = False
  if matching_indices:
    corner_images_match_count += 1
  else:
    corner_images_match_count = 0

  if corner_images_match_count == 0:
    last_corner_top_indices = logic.model.search_clip_image_corners()
  
  logic.append_history(last_corner_top_indices[shown * corner_images_match_count:shown * (corner_images_match_count + 1)].tolist())
  for i in range(4):
    last_corner_images[i] = logic.model.selected_corner_images[i]
  display_images()

def blacklist_shown_frames_shortcut():
  if shortcuts_disabled:
    return

  logic.blacklist_shown_frames()
  display_images()

def blacklist_shown_videos_shortcut():
  if shortcuts_disabled:
    return

  logic.blacklist_shown_videos()
  display_images()

viewing_full_image = False
def view_full_image_shortcut():
  global viewing_full_image

  if shortcuts_disabled:
    return

  if viewing_full_image:
    dpg.delete_item("full image tag")
    dpg.delete_item("full image texture tag")
    viewing_full_image = False
    show_all_images()
    hide_borders()
    return

  if len(logic.selected_images) <= 0:
    return
  
  hide_images(0)
  
  dkp_img = logic.get_resized_dpg_image(logic.selected_images[0], images_width, screen_height)
  with dpg.texture_registry():
    dpg.add_static_texture(width=images_width, height=screen_height, default_value=dkp_img, tag="full image texture tag")
  dpg.add_image("full image texture tag", tag="full image tag", pos=[0, 0], parent=window)
  hide_borders()
  viewing_full_image = True

def reset_scores_shortcut():
  logic.model.reset_scores()
  update_non_zero_scores_text()

def toggle_shortcuts_callback():
  set_shortcuts_disabled(not shortcuts_disabled)

def send_text_shortcut():
  text = dpg.get_value("text")
  # there is no send text functionality
  # send_text(text)

with dpg.texture_registry() as registry:
  for row in range(button_rows):
    for col in range(button_cols):
      dpg.add_static_texture(width=button_width, height=button_height, default_value=logic.get_dpg_image(row * button_cols + col), tag=get_texture_tag(row, col))
  
  default_corners = logic.get_dpg_image_corners(0)
  dpg.add_dynamic_texture(width=corner_width, height=corner_height, default_value=default_corners[0], tag="corner_tex_0")
  dpg.add_dynamic_texture(width=corner_width, height=corner_height, default_value=default_corners[1], tag="corner_tex_1")
  dpg.add_dynamic_texture(width=corner_width, height=corner_height, default_value=default_corners[2], tag="corner_tex_2")
  dpg.add_dynamic_texture(width=corner_width, height=corner_height, default_value=default_corners[3], tag="corner_tex_3")

  # create a white rectangle (with an alpha of 0.5) for yolo drawings
  background_pixel = np.array([[255, 255, 255, 127]], dtype=np.uint8)
  default_yolo = np.repeat(background_pixel, yolo_width * yolo_height, axis=0).flatten() / 255
  dpg.add_static_texture(width=yolo_width, height=yolo_height, default_value=default_yolo, tag="yolo_tex")

with dpg.item_handler_registry(tag="image click handler"):
  dpg.add_item_clicked_handler(callback=img_click_callback)

with dpg.item_handler_registry(tag="yolo click handler"):
  dpg.add_item_clicked_handler(callback=yolo_click_callback)

with dpg.handler_registry():
  dpg.add_key_press_handler(key=dpg.mvKey_Left, callback=go_back_shortcut)
  dpg.add_key_press_handler(key=dpg.mvKey_Right, callback=go_forward_shortcut)
  dpg.add_key_press_handler(key=dpg.mvKey_V, callback=show_video_shortcut)
  dpg.add_key_press_handler(key=dpg.mvKey_Return, callback=search_clip_shortcut)
  dpg.add_key_press_handler(key=dpg.mvKey_S, callback=score_search_shortcut)
  dpg.add_key_press_handler(key=dpg.mvKey_Control, callback=reset_scores_shortcut)
  dpg.add_key_press_handler(key=dpg.mvKey_I, callback=view_full_image_shortcut)
  dpg.add_key_press_handler(key=dpg.mvKey_Shift, callback=(lambda: focus_input_shortcut(True)))
  dpg.add_key_press_handler(key=dpg.mvKey_Q, callback=blacklist_shown_frames_shortcut)
  dpg.add_key_press_handler(key=dpg.mvKey_W, callback=blacklist_shown_videos_shortcut)
  dpg.add_key_press_handler(key=dpg.mvKey_C, callback=image_score_search_shortcut)

  dpg.add_key_press_handler(key=dpg.mvKey_F1, callback=(lambda: select_corner_shortcut(0)))
  dpg.add_key_press_handler(key=dpg.mvKey_F2, callback=(lambda: select_corner_shortcut(1)))
  dpg.add_key_press_handler(key=dpg.mvKey_F3, callback=(lambda: select_corner_shortcut(2)))
  dpg.add_key_press_handler(key=dpg.mvKey_F4, callback=(lambda: select_corner_shortcut(3)))

  dpg.add_key_press_handler(key=dpg.mvKey_F5, callback=(lambda: increase_corner_weight_shortcut(0)))
  dpg.add_key_press_handler(key=dpg.mvKey_F6, callback=(lambda: increase_corner_weight_shortcut(1)))
  dpg.add_key_press_handler(key=dpg.mvKey_F7, callback=(lambda: increase_corner_weight_shortcut(2)))
  dpg.add_key_press_handler(key=dpg.mvKey_F8, callback=(lambda: increase_corner_weight_shortcut(3)))

  dpg.add_key_press_handler(key=dpg.mvKey_F9, callback=increase_alpha_shortcut)
  dpg.add_key_press_handler(key=dpg.mvKey_F10, callback=decrease_alpha_shortcut)

  dpg.add_key_press_handler(key=dpg.mvKey_F11, callback=send_text_shortcut)
  dpg.add_key_press_handler(key=dpg.mvKey_F12, callback=(lambda: focus_input_shortcut(False)))

with dpg.window(label="Tool Window", width=tools_width, height=screen_height, no_collapse=True, no_resize=True, no_close=True, no_move=True, no_title_bar=True) as tools:
  dpg.add_input_text(tag="search")
  dpg.add_text(tag="selected", wrap=tools_width - 20)
  dpg.add_text(tag="non zero scores", wrap=tools_width - 20)
  dpg.add_text("alpha: 0.5", tag="alpha")
  dpg.add_text("")
  dpg.add_text("Shortcuts ENABLED", tag="shortcut status")
  dpg.add_button(label="Toggle Shortcuts", callback=toggle_shortcuts_callback)
  dpg.add_text("")
  dpg.add_text("[Left] Go Back [d]")
  dpg.add_text("[Right] Go Forward [d]")
  dpg.add_text("[V] Show Video [d]")
  dpg.add_text("[Return] Prompt Search")
  dpg.add_text("[S] Score Search [d]")
  dpg.add_text("[Ctrl] Reset Scores")
  dpg.add_text("[I] Toggle Big Image [d]")
  dpg.add_text("[F12] Focus Prompt")
  dpg.add_text("[LShift] Focus Empty Prompt")
  dpg.add_text("[Q] Blacklist Frames [d]")
  dpg.add_text("[W] Blacklist Videos [d]")
  dpg.add_text("[C] Corner Search [d]")
  dpg.add_text("[F1-F4] Select Corner")
  dpg.add_text("[F5-F8] Increase Corner Weight", wrap=tools_width - 20)
  dpg.add_text("[F9] Increase Alpha")
  dpg.add_text("[F10] Decrease Alpha")

  dpg.add_text("")

  with dpg.group(horizontal=True):
    dpg.add_input_text(tag="search_0")
    dpg.add_text("x", tag="corner_weight_extra_0")
    dpg.add_text("1", tag="corner_weight_0")

  with dpg.group(horizontal=True):
    dpg.add_input_text(tag="search_1")
    dpg.add_text("x", tag="corner_weight_extra_1")
    dpg.add_text("1", tag="corner_weight_1")

  with dpg.group(horizontal=True):
    dpg.add_input_text(tag="search_2")
    dpg.add_text("x", tag="corner_weight_extra_2")
    dpg.add_text("1", tag="corner_weight_2")

  with dpg.group(horizontal=True):
    dpg.add_input_text(tag="search_3")
    dpg.add_text("x", tag="corner_weight_extra_3")
    dpg.add_text("1", tag="corner_weight_3")

  corner_bottom_offset = 300
  dpg.add_image("corner_tex_0", tag="corner_img_0", pos=[0, screen_height - corner_bottom_offset])
  dpg.add_image("corner_tex_1", tag="corner_img_1", pos=[0, screen_height - corner_bottom_offset + corner_height])
  dpg.add_image("corner_tex_2", tag="corner_img_2", pos=[corner_width, screen_height - corner_bottom_offset])
  dpg.add_image("corner_tex_3", tag="corner_img_3", pos=[corner_width, screen_height - corner_bottom_offset + corner_height])

  # yolo drawing img
  dpg.add_image("yolo_tex", tag="yolo_img", pos=[0, yolo_y_pos])
  dpg.bind_item_handler_registry("yolo_img", "yolo click handler")

rect_ids = []
with dpg.window(label="Image Window", width=images_width, height=screen_height, no_collapse=True, no_resize=True, no_close=True, no_move=True, no_title_bar=True, pos=[tools_width, 0]) as window:
  for row in range(button_rows):
    for col in range(button_cols):
      x, y = get_img_position(row, col)
      rectX, rectY = get_rect_position(row, col)

      rectId = dpg.draw_rectangle([rectX, rectY], [rectX + button_width, rectY + button_height], tag=get_border_tag(row, col), show=False, thickness=3, color=[255,255,150])
      rect_ids.append(rectId)

      img_tag = get_img_tag(row, col)
      img_data = {
         "pos": [row, col],
         "rectId": rectId,
         "imgIdx": row * button_cols + col
      }
      dpg.add_image(get_texture_tag(row, col), tag=img_tag, pos=[x, y], user_data=img_data)
      dpg.bind_item_handler_registry(img_tag, "image click handler")

update_non_zero_scores_text()
dpg.show_viewport()
dpg.toggle_viewport_fullscreen()
hide_input()
dpg.start_dearpygui()
dpg.destroy_context()
