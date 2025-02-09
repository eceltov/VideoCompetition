import open_clip
import pickle
import torch
import torch.nn.functional as F
import numpy as np
import time
from PIL import Image
import math

class Model:
  # init globals and load model
  def __init__(self) -> None:
    self.device = "cuda"
    self.print_debug = False

    self.frame_size = [682, 384]

    self.model, _, self.preprocess = open_clip.create_model_and_transforms('ViT-B-32',
      pretrained='laion2b_s34b_b79k', device=self.device)
    self.tokenizer = open_clip.get_tokenizer('ViT-B-32')
    self.features = self.load_clip().to(self.device)
    self.corner_features = [corner.to(self.device) for corner in self.load_clip_quarters()]
    self.detection_boxes = self.load_detection_boxes()
    self.box_features = self.load_box_features()

    self.model.eval()

    print("feature length:", self.features.shape[1], self.features.dtype)

    self.alpha = 0.5

    self.corner_weights = [1, 1, 1, 1]
    self.selected_corner_images = [0, 0, 0, 0]

    self.image_count = self.features.shape[0]
    self.scores = np.empty(self.image_count)
    self.scores = torch.tensor(self.scores)
    self.scores.fill_(1 / self.image_count)

  def reset_scores(self):
    self.scores.fill_(1 / self.image_count)

  def L2S(self, feature1D, subtrahend1D):
    return torch.norm(feature1D - subtrahend1D)
  
  def L2SBulk(self, features2D, subtrahend1D):
    return torch.norm(features2D - subtrahend1D, dim=1)
  
  def L2SSuperBulk(self, features3D, subtrahends3D):
    return torch.norm(features3D - subtrahends3D, dim=2)

  def normalize_scores_max(self):
    self.scores /= self.scores.max()

  def update_scores(self, display, likeID):
    startTime = time.time()
    
    # create a tensor of display features and duplicate them self.image_count times
    display_features = torch.stack([self.features[i] for i in display])
    display_features = torch.unsqueeze(display_features, 0).expand(self.image_count, -1, -1)
    # reshape image features
    image_features = torch.unsqueeze(self.features, 1)

    preprocess = time.time()

    # compute norms of image feature and liked feature pairs
    PFs = torch.exp(-self.L2SBulk(self.features, self.features[likeID]) / self.alpha)

    img_liked = time.time()

    # subtract image features from the duplicated display features and compute the norm
    NFs = torch.exp(-self.L2SSuperBulk(display_features, image_features) / self.alpha).sum(1)

    img_display = time.time()

    # move the scores to the cpu for faster access
    self.scores = (self.scores.to(self.device) * PFs / NFs).to("cpu")

    self.normalize_scores_max()
    endTime = time.time()

    if self.print_debug:
      print("preprocess:", preprocess - startTime, "img/liked norms:", img_liked - preprocess, "img/display norms:", img_display - img_liked, "postprocess:", endTime - img_display, "total:", endTime - startTime)
    else:
      print("Update Scores:", endTime - startTime)

  def get_top_score_indices(self, count):
    top_k = np.argsort(-self.scores)
    return top_k[:count]
  
  def non_zero_scores_count(self):
    return (self.image_count - (self.scores < 0.000001).sum()).item()

  def set_scores_to_zero(self, indices):
    for i in indices:
      self.scores[i] = 0

  def search_clip(self, text: str) -> list[int]:
    query = self.tokenizer(text).to(self.device)

    with torch.no_grad(), torch.cuda.amp.autocast():
      text_features = self.model.encode_text(query)

      distances = 1 - (F.normalize(text_features) @ F.normalize(self.features).T)

      sorted_indices = torch.argsort(distances)[0]

    return sorted_indices
  
  # searches in detection boxes and ignores whole features
  def search_clip_boxes_only(self, source_box, text: str) -> list[int]:
    box_features_indices = []
    selected_box_features = []

    for frame_idx in range(self.image_count):
      box_idx = self.detection_boxes.get_best_IoU_box_idx(source_box, self.frame_size, frame_idx)
      # only use box features
      if box_idx != -1:
        box_features_indices.append(frame_idx)
        selected_box_features.append(self.box_features[frame_idx][box_idx].view(1, self.features.shape[1]))

    # create tensor from collected box features
    selected_box_features = torch.concat(selected_box_features).to(self.device)

    query = self.tokenizer(text).to(self.device)

    with torch.no_grad(), torch.cuda.amp.autocast():
      text_features = self.model.encode_text(query)

      distances = 1 - (F.normalize(text_features) @ F.normalize(selected_box_features).T)

      sorted_indices = torch.argsort(distances)[0]

    # convert feature rankings to feature indices
    projected_indices = torch.zeros(sorted_indices.shape, dtype=sorted_indices.dtype)
    for i in range(len(box_features_indices)):
      projected_indices[i] = box_features_indices[sorted_indices[i]]

    print(f"<box indices: {len(box_features_indices)}/{self.image_count}>", end=" ")

    return projected_indices
  
  # searches in detection boxes and whole features
  def search_clip_boxes(self, source_box, text: str) -> list[int]:
    box_features_count = 0

    # construct a feature tensor for the given source box
    selected_box_features = torch.zeros(self.features.shape, dtype=self.features.dtype)
    for frame_idx in range(self.image_count):
      box_idx = self.detection_boxes.get_best_IoU_box_idx(source_box, self.frame_size, frame_idx)
      # use whole features if there is no localization candidate
      if box_idx == -1:
        selected_box_features[frame_idx] = self.features[frame_idx]
      # use box features if they have bigger IoU than the whole image box
      else:
        box_features_count += 1
        selected_box_features[frame_idx] = self.box_features[frame_idx][box_idx]

    selected_box_features = selected_box_features.to(self.device)

    query = self.tokenizer(text).to(self.device)

    with torch.no_grad(), torch.cuda.amp.autocast():
      text_features = self.model.encode_text(query)

      distances = 1 - (F.normalize(text_features) @ F.normalize(selected_box_features).T)

      sorted_indices = torch.argsort(distances)[0]

    print(f"<box indices: {box_features_count}/{self.image_count}>", end=" ")

    return sorted_indices
  
  # searches in each corner using text and aggregates the rankings
  def search_clip_corners(self, texts: list[str]) -> list[int]:
    queries = []
    for i in range(4):
      if texts[i] != None and len(texts[i]) > 0:
        queries.append(self.tokenizer(texts[i]).to(self.device))
      else:
        queries.append(None)

    with torch.no_grad(), torch.cuda.amp.autocast():
      text_features = []
      for query in queries:
        if query != None:
          text_features.append(self.model.encode_text(query))
        else:
          text_features.append(None)

      corner_distances = []
      for i in range(4):
        if text_features[i] != None:
          corner_distances.append((1 - (F.normalize(text_features[i]) @ F.normalize(self.corner_features[i]).T)) * self.corner_weights[i])

      total_distances = corner_distances[0]
      for i in range(1, len(corner_distances)):
        total_distances += corner_distances[i]

      sorted_indices = torch.argsort(total_distances)[0]

    return sorted_indices
  
  # searches in each corner using image similarity and aggregates the rankings
  def search_clip_image_corners(self):
    corner_distances = []
    for i in range(4):
      corner_feature = self.features[self.selected_corner_images[i]]
      distances = 1 - (F.normalize(torch.unsqueeze(corner_feature, 0)) @ F.normalize(self.features).T)
      corner_distances.append(distances)

    sortings = [torch.argsort(corner_distances[i])[0].to("cpu") for i in range(4)]
    scores = np.zeros(self.image_count)

    for i in range(self.image_count):
      for corner_idx in range(4):
        scores[sortings[corner_idx][i]] += i

    return np.argsort(scores)

  # returns an image segment for every corner
  def get_image_corners(self, filename):
    image = Image.open(filename)
    width, height = image.size

    boundaries = [
      (0, 0, width // 2, height // 2), # left upper
      (0, height // 2, width // 2, height), # left lower
      (width // 2, 0, width, height // 2), # right upper
      (width // 2, height // 2, width, height), # right lower
    ]

    corners = [image.crop(boundary) for boundary in boundaries]
    return corners
  
  def load_clip_quarters(self):
    with open('features/cornerFeatures.pickle', 'rb') as handle:
      return pickle.load(handle)
    
  def load_clip(self):
    with open('features/wholeFeatures.pickle', 'rb') as handle:
      # there is only one segment, return it instead
      return pickle.load(handle)[0]

  def load_detection_boxes(self):
    with open('features/detectionBoxes.pickle', 'rb') as handle:
      return pickle.load(handle)
    
  def load_box_features(self):
    with open('features/boxFeatures.pickle', 'rb') as handle:
      # a list of 2D tensors with rows for each box and columns as features
      return pickle.load(handle)
