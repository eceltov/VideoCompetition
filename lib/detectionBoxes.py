class DetectionBoxes:
  def __init__(self, boxes, counts):
    self.boxes = boxes
    self.counts = counts

  def get_box(self, frame_idx, box_idx):
    box = self.boxes[frame_idx][box_idx]
    return box.tolist()
