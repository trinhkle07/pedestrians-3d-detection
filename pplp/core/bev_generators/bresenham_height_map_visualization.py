#!/usr/bin/python

import matplotlib.pyplot as plt
import numpy as np
import os

# plt.ioff()

def main():
    height_map_dir = "/home/trinhle/Downloads/cam_lidar_mocap_test_set/bev_vis"
    height_map_viz_dir = "/home/trinhle/Downloads/cam_lidar_mocap_test_set/bev_vis/viz"
    if not os.path.exists(height_map_viz_dir):
        os.mkdir(height_map_viz_dir)

    for (dirpath, dirnames, filenames) in os.walk(height_map_dir):
        for filename in filenames:
            try:
                print(filename)
                arr = np.loadtxt(dirpath + "/" + filename)
                plt.imshow(arr)
                # fig = plt.gcf()
                # fig.set_size_inches((8.5, 11), forward=False)
                # plt.show()
                plt.savefig(height_map_viz_dir + "/" + filename[:-4])
            except Exception:
                pass

        break # only loop the first level dir


if __name__ == "__main__":
    main()
