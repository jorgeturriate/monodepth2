# Copyright Niantic 2019. Patent Pending. All rights reserved.
#
# This software is licensed under the terms of the Monodepth2 licence
# which allows for non-commercial use only, the full terms of which are made
# available in the LICENSE file.

from __future__ import absolute_import, division, print_function

import os
import skimage.transform
import numpy as np
import PIL.Image as pil

from kitti_utils import generate_depth_map
from .mono_dataset_mid import MonoDataset



class MidAirDataset(MonoDataset):
    def __init__(self, *args, **kwargs):
        super(MidAirDataset, self).__init__(*args, **kwargs)
        
        self.full_res_shape = (1024, 1024)  # width, height
        
        # Set fx, fy, cx, cy based on resize size
        # fx = cx = 0.5 * width
        # fy = cy = 0.5 * height
        
        self.K = np.array([
            [0.5,  0, 0.5, 0],
            [0, 0.5, 0.5, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ], dtype=np.float32)


    def get_color(self, folder, frame_index, side, do_flip):
        # MidAir has only monocular RGB data
        color_path = os.path.join(self.data_path, folder, f"{frame_index:06d}{self.img_ext}")
        color = self.loader(color_path)

        if do_flip:
            color = color.transpose(pil.FLIP_LEFT_RIGHT)

        return color
    
    def check_depth(self):
        line = self.filenames[0].split()
        depth_name = line[1]
        frame_index = int(line[2])

        depth_filename = os.path.join(
            self.data_path,
            depth_name,
            f"{frame_index:06d}.PNG")

        return os.path.isfile(depth_filename)


    def get_depth(self, folder, frame_index, side, do_flip):
        # MidAir stores depth as 16-bit PNG (as disparity), encoded from float16
        depth_path = os.path.join(self.data_path, folder, f"{frame_index:06d}.PNG")
        depth_png = pil.open(depth_path)
        depth_png = np.array(depth_png, dtype=np.uint16)
        # Decode float16 bitstream from uint16 using view
        depth_float16 = depth_png.view(dtype=np.float16)
        # Convert to actual depth values using the same formula: depth = 512. / disparity
        disparity = depth_float16.astype(np.float32)
        disparity[disparity == 0] = 0.01  # avoid division by 0
        depth = 512. / disparity

        depth = skimage.transform.resize(
            depth, self.full_res_shape[::-1], order=0, preserve_range=True, mode='constant')

        if do_flip:
            depth = np.fliplr(depth)

        return depth
