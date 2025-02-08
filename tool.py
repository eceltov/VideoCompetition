# Tool template developed by Z. Vopalkova, allowed to use only for the competition in the course
# Required update - new format of image file name, for submission filename and time are required (code will be provided later)
# Find data for submission here http://siret.ms.mff.cuni.cz/lokoc/metadata.csv

import tkinter as tk
from tkinter import ttk

import requests
from PIL import Image, ImageTk
import os
import pickle
import time

from lib.model import Model

model = Model()

image_cols = 6
image_rows = 8
shown = image_cols * image_rows
url = "https://siret.ms.mff.cuni.cz/lokoc/VBSEval/EndPoint.php"
dataset_path = "D:\\school\\videa\\seaPics"

# get images address
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

root = tk.Tk()
root.title("Searcher")
root.wm_attributes('-fullscreen', 'true')

image_size = (int(root.winfo_screenwidth() / image_cols) - 50, int(root.winfo_screenheight() / image_rows) - 6)

# contains top_result lists of previous actions
history = []
historyImages = []
history_idx = 0

# frames that did not fit on the screen
not_shown_video_frames = []

images_buttons = []
selected_images = []

def savePreprocessImages(filenames, image_size):
    # preprocess images to speed up the search engine
    image_cache = []
    for filename in filenames:
        image_cache.append(Image.open(filename).resize(image_size))

    with open('image_cache.pickle', 'wb') as handle:
        pickle.dump(image_cache, handle, protocol=pickle.HIGHEST_PROTOCOL)

def loadPreprocessedImages():
    with open('image_cache.pickle', 'rb') as handle:
        return pickle.load(handle)

def update_selected_images_text():
    if len(selected_images) > 0:
        text_index.config(text="Selected images: " + str(selected_images))
    else:
        text_index.config(text="No image selected.")

def hide_borders():
    global selected_images
    for button in images_buttons:
        button.config(bg="black")
    selected_images = []
    update_selected_images_text()

def append_history(indices):
    global history
    global history_idx
    global historyImages

    # remove 'future' searches if a new forward branch is made
    history = history[:history_idx + 1]
    historyImages = historyImages[:history_idx + 1]
    history_idx += 1

    # append image indices and create new images
    history.append(indices)

    historyImages.append([ImageTk.PhotoImage(Image.open(filenames[indices[i]]).resize(image_size)) for i in range(len(indices))])


    #historyImages.append([ImageTk.PhotoImage(image_cache[indices[i]]) for i in range(len(indices))])

def search_clip(text):
    global history
    global history_idx
    global historyImages
    global search_bar

    search_bar.focus_set()

    startTime = time.time()
    print(text, end=" ")
    top_result = model.search_clip(text)

    append_history(top_result[:shown].tolist())

    display_images()

    print(time.time() - startTime)

def show_button(idx):
    images_buttons[idx].grid(row=(idx // image_cols), column=(idx % image_cols), sticky=tk.W)

def hide_button(idx):
    images_buttons[idx].grid_forget()

def display_images():
    global history
    global history_idx
    global historyImages

    # top result - sorted score (position)
    top_result = history[history_idx]

    for i in range(len(top_result)):
        show_button(i)
        images_buttons[i].configure(image=historyImages[history_idx][i], text=filenames[top_result[i]],
                                    command=(lambda j=i: on_click(j, top_result[j])))
        
    for i in range(len(top_result), shown):
        hide_button(i)
    hide_borders()

def on_click(btnIndex, imgIndex):
    if images_buttons[btnIndex].cget("bg") == "yellow":
        images_buttons[btnIndex].config(bg="black")
        selected_images.remove(imgIndex)
    else:
        images_buttons[btnIndex].config(bg="yellow")
        selected_images.append(imgIndex)
        #print("selected", filenames[imgIndex])
    update_selected_images_text()

def go_back():
    global history_idx

    if history_idx <= 0:
        return
    
    history_idx -= 1
    display_images()

def go_forward():
    global history
    global history_idx

    if history_idx < len(history) - 1:
        history_idx += 1
        display_images()

def show_video():
    global not_shown_video_frames

    if len(selected_images) <= 0:
        if len(not_shown_video_frames) > 0:
            append_history(not_shown_video_frames[:shown])
            not_shown_video_frames = not_shown_video_frames[shown:]
            display_images()
        return
    
    idx = selected_images[0]
    path = frame_idx_to_frame_path_map[idx]
    dirpath = os.path.dirname(path)
    frames = video_to_frame_indices_map[dirpath]
    append_history(frames[:shown])
    not_shown_video_frames = frames[shown:]
    display_images()

def score_search():
    if len(selected_images) <= 0:
        return

    likeID = selected_images[0]
    model.update_scores(history[-1], likeID, 0.5)
    append_history(model.get_top_score_indices(shown)[:shown].tolist())
    display_images()

def on_double_click():
    hide_borders()


def close_win(e):
    root.destroy()

def send_result():
    pass
    #key_i = (selected_images[0][-9:])[:5]
    #my_obj = {'team': "name", 'item': key_i}

    #x = requests.get(url=url, params=my_obj, verify=False)
    #print(x.text)

#image_cache = loadPreprocessedImages()

# create window
window = ttk.Panedwindow(root, orient=tk.HORIZONTAL)
window.pack(fill=tk.BOTH, expand=True)

# create frames
search_bar = ttk.Frame(window, width=root.winfo_screenwidth() / 4, height=root.winfo_screenheight(), relief=tk.SUNKEN)
result_frame = ttk.Frame(window, width=(3 * root.winfo_screenwidth()) / 4, height=root.winfo_screenheight(),
                         relief=tk.SUNKEN)
window.add(search_bar, weight=1)
window.add(result_frame, weight=4)

# add text input
def clear_and_focus_input():
    global text_input
    text_input.delete(0, tk.END)
    text_input.focus_set()

tk.Label(search_bar, text="Text query:").pack(side=tk.TOP, pady=5)
text_input = tk.Entry(search_bar, bd=3, width=32)
text_input.bind("<Return>", (lambda event: search_clip(text_input.get())))
root.bind("f", (lambda event: text_input.focus_set()))
root.bind("<Control-f>", (lambda event: clear_and_focus_input()))
#root.bind("<Control-f>", (lambda event: search_bar.focus_set()))
text_input.pack(side=tk.TOP, pady=5)

# add search buttons
clip_button = tk.Button(search_bar, text="Search Clip", command=(lambda: search_clip(text_input.get())))
clip_button.pack(side=tk.TOP)

# add history buttons
go_back_button = tk.Button(search_bar, text="Back", command=(lambda: go_back()))
go_back_button.pack(side=tk.TOP)
root.bind('<Left>', lambda e: go_back())

go_forward_button = tk.Button(search_bar, text="Forward", command=(lambda: go_forward()))
go_forward_button.pack(side=tk.TOP)
root.bind('<Right>', lambda e: go_forward())

# show video button
show_video_button = tk.Button(search_bar, text="Show Video", command=(lambda: show_video()))
show_video_button.pack(side=tk.TOP)
root.bind('v', lambda e: show_video())

# score search
root.bind('s', lambda e: score_search())
root.bind('<Control-s>', lambda e: model.reset_scores())

# add info labels
tk.Label(search_bar, text="Find index (1-11870):").pack(side=tk.TOP, pady=10)
text_index = tk.Label(search_bar, text="Last selected image: ")
text_index.pack(side=tk.TOP, pady=5)

# sending select result
send_result_b = tk.Button(search_bar, text="Send selected index", command=(lambda: send_result()))
send_result_b.pack(side=tk.TOP, pady=5)
# set control-v to set result
#root.bind('<Control-v>', lambda e: send_result())

# set images
history.append(range(shown))
# load images

historyImages.append([ImageTk.PhotoImage(Image.open(filenames[i]).resize(image_size)) for i in range(shown)])
#historyImages.append([ImageTk.PhotoImage(image_cache[i]) for i in range(shown)])
for s in range(shown):
    # create button
    images_buttons.append(tk.Button(result_frame, bg="black", bd=2, text=filenames[s], image=historyImages[history_idx][s],
                                    command=(lambda j=s: on_click(j, j))))
    # set position of button
    images_buttons[s].grid(row=(s // image_cols), column=(s % image_cols), sticky=tk.W)
    # set double click to reset marking of images
    images_buttons[s].bind('<Double-1>', lambda event: on_double_click())

# set escape as exit
root.bind('<Escape>', lambda e: close_win(e))

root.mainloop()