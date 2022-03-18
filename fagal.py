import os
import shutil
import numpy as np

from scipy.ndimage import zoom
from skimage.data import camera
from scipy.spatial.distance import cdist


def duration_to_alignment(in_duration):
  total_len = np.sum(in_duration)
  num_chars = len(in_duration)

  attention = np.zeros(shape=(num_chars,total_len),dtype=np.float32)
  y_offset = 0

  for duration_idx, duration_val in enumerate(in_duration):
    for y_val in range(0,duration_val):
      attention[duration_idx][y_offset + y_val] = 1.0
    
    y_offset += duration_val
  
  return attention


def rescale_alignment(in_alignment,in_targcharlen):
  current_x = in_alignment.shape[0]
  x_ratio = in_targcharlen / current_x
  pivot_points = []
  
  zoomed = zoom(in_alignment,(x_ratio,1.0),mode="nearest")

  for x_v in range(0,zoomed.shape[0]):
    for y_v in range(0,zoomed.shape[1]):
      val = zoomed[x_v][y_v]
      if val < 0.5:
        val = 0.0
      else:
        val = 1.0
        pivot_points.append( (x_v,y_v) )

      zoomed[x_v][y_v] = val
      
  
  if zoomed.shape[0] != in_targcharlen:
    print("Zooming didn't rshape well, explicitly reshaping")
    zoomed.resize((in_targcharlen,in_alignment.shape[1]))

  return zoomed, pivot_points

    


def compensate_duration_mel(in_durs,in_mel_len,in_text_len,in_audiofn):
    durations_len = np.sum(in_durs)
    durations_n = in_durs
    
    if in_mel_len > durations_len:
        duration_n[-1] += in_mel_len - durations_len
    else:
        if durations_len > in_mel_len:
            diff = durations_len - in_mel_len
            found = False
            for r in reversed(range(len(duration_n) - 1)):
                if diff == 0:
                    found = True
                    break
                    
                if duration_n[r] >= diff:
                    duration_n[r] -= diff
                    found = True
                    break
                else:
                    diff -= duration_n[r]
                    duration_n[r] = 0
                    
            if not found:
                raise RuntimeError(f"Oh nonono MFABros what happened!!!!!!!????? (Failed to compensate for diff in durations and mel len for file {in_audiofn})")
                
                
    return durations_n


def durations_to_mask(in_durs,in_mel_len,in_text_len,in_audiofn,in_loose):
    duration_arr = compensate_duration_mel(in_durs,in_mel_len,in_text_len,in_audiofn)
    align = duration_to_alignment(duration_arr)
        
    if align.shape[0] != id_true_size:
        align, points = rescale_alignment(align,id_true_size)
    else:
        points = get_pivot_points(align)
    
    mask = create_guided(align,points,in_loose)
    return mask
    
