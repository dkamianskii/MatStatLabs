import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np

def save_projections_for_video(cur_data, time_start, dt, window_big, save_folder):
    i = 0
    while i < cur_data.shape[2]:
        tmp = cur_data[:, :, i]
        fig, ax = plt.subplots(subplot_kw={'xticks': [2, 4, 6, 8, 10, 12, 14, 16],
                                           'yticks': [2, 4, 6, 8, 10, 12, 14, 16]})
        im = ax.imshow(tmp, origin='lower')
        ax.set_title(f'{np.round(time_start + dt * i, decimals=5)} ms')
        divider = make_axes_locatable(ax)
        cax = divider.append_axes('right', size='5%', pad=0.05)
        plt.colorbar(im, cax=cax, orientation='vertical')
        file_name = f'{save_folder}/plasm_proj_{i}.png'
        plt.savefig(file_name, format="png")
        plt.close(fig)
        i += window_big


def save_centers_for_video(inds_of_diffs_rot, save_folder):
    i = 0
    for ind in inds_of_diffs_rot:
        fig, ax = plt.subplots()
        ax.scatter(ind[0][0], ind[0][1])
        ax.grid()
        ax.set_xlim([0, 16])
        ax.set_ylim([0, 16])
        file_name = f'{save_folder}/center_mass_img_{i}.png'
        plt.savefig(file_name, format="png")
        plt.close(fig)
        i += 1