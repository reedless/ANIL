from torch.utils.data import Dataset

import cv2
import numpy as np

class ListDataset(Dataset):
    def __init__(self, data):
        self.data = data
    def __getitem__(self, i):
        return self.data[i]
    def __len__(self):
        return len(self.data)


def create_image(image, accumulate_strokes=True, output_dims=(224, 224)):
    min_x, max_x, min_y, max_y = _get_bounds(image)
    dims = (50 + max_x - min_x, 50 + max_y - min_y)

    pixels = np.zeros((dims[1], dims[0]))
    strokes = []
    
    prev_x = 25 - min_x 
    prev_y = 25 - min_y
    
    move = True
    
    for point in image:
        if move:
            # start of stroke, move to new initial point
            prev_x += point[0]
            prev_y += point[1]
            move = False
        else:
            # middle or end of stroke, add line to pixels
            curr_x = prev_x + point[0]
            curr_y = prev_y + point[1]
            line = _get_line(prev_x, prev_y, curr_x, curr_y)
            for x, y in line:
                pixels[round(y), x] = 1
            prev_x = curr_x
            prev_y = curr_y
            
            if point[2] == 1:
                # if end of stroke, add current stroke to strokes and move to next point
                move = True
                strokes.append(pixels.copy())

                if not accumulate_strokes:
                    # reset picture
                    pixels = np.zeros((dims[1], dims[0]))
    
    strokes = np.stack(strokes, axis=-1).astype(np.int16)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2,2))
    strokes = cv2.dilate(strokes, kernel, iterations=2)
    strokes = strokes.astype(np.int16)
    
    return cv2.resize(strokes, output_dims)


def _get_bounds(data):
    """Return bounds of data."""
    min_x = 0
    max_x = 0
    min_y = 0
    max_y = 0

    abs_x = 0
    abs_y = 0
    
    for i in range(len(data)):
        x = data[i, 0]
        y = data[i, 1]
        abs_x += x
        abs_y += y
        min_x = min(min_x, abs_x)
        min_y = min(min_y, abs_y)
        max_x = max(max_x, abs_x)
        max_y = max(max_y, abs_y)

    return (min_x, max_x, min_y, max_y)


def _get_line(x1, y1, x2, y2):
  points = []
  issteep = abs(y2-y1) > abs(x2-x1)
  if issteep:
      x1, y1 = y1, x1
      x2, y2 = y2, x2
  rev = False
  if x1 > x2:
      x1, x2 = x2, x1
      y1, y2 = y2, y1
      rev = True
  deltax = x2 - x1
  deltay = abs(y2-y1)
  error = int(deltax / 2)
  y = y1
  ystep = None
  if y1 < y2:
      ystep = 1
  else:
      ystep = -1
  for x in range(x1, x2 + 1):
      if issteep:
          points.append((y, x))
      else:
          points.append((x, y))
      error -= deltay
      if error < 0:
          y += ystep
          error += deltax
  # Reverse the list if the coordinates were reversed
  if rev:
      points.reverse()
  return points
