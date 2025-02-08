class DetectionBoxes:
  def __init__(self, boxes, counts):
    self.boxes = boxes
    self.counts = counts

  def get_box(self, frame_idx, box_idx):
    box = self.boxes[frame_idx][box_idx]
    return box.tolist()

  def get_box_IoU(self, source_box, target_box):
    # swap source_box coords if the first point is positioned after the second one
    if source_box[0] >= source_box[2] and source_box[1] >= source_box[3]:
      # do not overwrite the source list
      source_box = [
        source_box[2],
        source_box[3],
        source_box[0],
        source_box[1],
      ]

    assert source_box[0] < source_box[2]
    assert source_box[1] < source_box[3]
    assert target_box[0] < target_box[2]
    assert target_box[1] < target_box[3]

    # compute intersection area
    x_left = max(source_box[0], target_box[0])
    y_top = max(source_box[1], target_box[1])
    x_right = min(source_box[2], target_box[2])
    y_bottom = min(source_box[3], target_box[3])

    if x_right < x_left or y_bottom < y_top:
        return 0.0

    intersection_area = (x_right - x_left) * (y_bottom - y_top)

    # compute rectangle areas
    source_box_area = (source_box[2] - source_box[0]) * (source_box[3] - source_box[1])
    target_box_area = (target_box[2] - target_box[0]) * (target_box[3] - target_box[1])

    iou = intersection_area / float(source_box_area + target_box_area - intersection_area)
    return iou

  # Returns the index of the box with the highest IoU with the source box.
  # In case there are no boxes or the best box IoU is lower than the IoU with the whole frame, -1 is returned instead.
  # -1 signals that whole frame features should be used rather than localized ones.
  def get_best_IoU_box_idx(self, source_box, frame_size, frame_idx):
    # return -1 if there is no box
    if self.counts[frame_idx] == 0:
      return -1
    
    whole_frame_box = [0, 0, frame_size[0], frame_size[1]]

    # the default best IoU and box index is of the whole frame
    best_IoU = self.get_box_IoU(source_box, whole_frame_box)
    best_box_idx = -1
    print("whole:", best_IoU)
    for box_idx in range(self.counts[frame_idx]):
      IoU = self.get_box_IoU(source_box, self.get_box(frame_idx, box_idx))
      print(box_idx, IoU)
      if IoU > best_IoU:
        best_IoU = IoU
        best_box_idx = box_idx

    return best_box_idx
