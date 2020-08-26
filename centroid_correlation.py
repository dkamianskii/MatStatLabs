import scipy.io as isc
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import make_video as video
from mpl_toolkits.axes_grid1 import make_axes_locatable
import pandas as pd
interval = [135.0, 175.0]
angle_limit = -89.995  # const for vertical movement


def get_data_from_file():
    all_variables = isc.loadmat('37000_SPD16x16.mat')
    data_array = np.rot90(all_variables.get("sign_bb"), 2)
    q_size = data_array.shape
    return data_array, all_variables, q_size


def prepare_extra_data(all_variables, main_matrix):
    data = all_variables.get("Data")
    sign_bb = np.array(all_variables.get("sign_bb"))

    dt = data[0][1][0][0] * 1.0e-3
    t_start = data[1][1][0][0]
    t_end = t_start + sign_bb.shape[2] * dt

    t_s = int((interval[0] - float(data[1, 1])) / dt)
    t_e = int((interval[1] - float(data[1, 1])) / dt)
    t = slice(t_s, t_e)
    matrix_low = main_matrix[:8, ...]
    matrix_up = main_matrix[8:, ...]

    return t, t_start, t_end, dt, matrix_up, matrix_low


def luminosity_plot(up_data, low_data, t, t_s):
    # The whole luminosity
    grid = np.linspace(t_s, t_s + dt * len(up_sum_data), len(up_sum_data))
    sns.lineplot(grid, up_data, label='Север', linewidth=1)
    sns.lineplot(grid, low_data, label='Юг', linewidth=1)
    plt.title('Суммарная светимость')
    plt.xlabel('t, ms')
    plt.ylabel('luminosity values')
    plt.legend()
    plt.savefig('luminosity_whole.png', format='png')
    plt.show()

    # Common luminosity on interval
    grid = np.linspace(interval[0], interval[0] + dt * len(up_sum_data[t]), len(up_sum_data[t]))
    sns.lineplot(grid, up_data[t], label='Север', linewidth=1)
    sns.lineplot(grid, low_data[t], label='Юг', linewidth=1)
    plt.title('Суммарная светимость на заданном интервале')
    plt.xlabel('t, ms')
    plt.ylabel('luminosity values')
    plt.legend()
    plt.savefig('luminosity.png', format='png')
    plt.show()


def correlation_coefficient(main_matrices):
    cur_data = main_matrices[:, :, t]
    window = int(1 / dt)
    cur_data_up = main_matrices[8:, :, t].sum(axis=(0, 1))
    cur_data_down = main_matrices[:8, :, t].sum(axis=(0, 1))
    correlations = [np.corrcoef(cur_data_up[i:i + window], cur_data_down[i:i + window])[0, 1]
                    for i in range(cur_data.shape[-1] - window)]
    n = len(correlations)
    x_grid = np.linspace(interval[0], interval[0] + dt * n, n)
    plt.plot(x_grid, correlations, color='blue', linewidth=1)
    plt.title("Коэффициент корреляции между севером и югом")
    plt.ylabel('y')
    plt.xlabel('t, ms')
    plt.savefig('correlation.png', format='png')
    plt.show()
    return correlations, x_grid


def centroid(main_matrices):
    cur_data = main_matrices[:, :, t]
    window = int(1 / dt)
    cur_data_up = main_matrices[8:, :, t].sum(axis=(0, 1))
    cur_data_down = main_matrices[:8, :, t].sum(axis=(0, 1))
    c_mass = [np.mean(np.array((np.mean(cur_data_up[i:i + window]), np.mean(cur_data_down[i:i + window]))))
              for i in range(cur_data.shape[-1] - window)]
    n = len(c_mass)
    x_grid = np.linspace(interval[0], interval[0] + dt * n, n)
    sns.lineplot(x_grid, c_mass, linewidth=1, color='deepskyblue', label='Траектория центра масс')
    plt.title('Движение центра масс без нормирования данных')
    plt.xlabel('t, ms')
    plt.ylabel('luminosity values')
    plt.legend()
    plt.savefig('centroid_wo_norm.png', format='png')
    plt.show()
    return c_mass


def centroid_and_correlation(correlations, normalized_df, index_start, index_end):
    # index_start - beginning of time window
    # index_end - end of time window
    n = len(correlations)
    x_grid = np.linspace(interval[0], interval[0] + dt * n, n)
    plt.plot(x_grid, correlations, linewidth=1, color='blue', label='Коэффициент корреляции')
    sns.lineplot(x_grid, normalized_df['y'], linewidth=1, color='deepskyblue', label='Траектория центра масс')
    plt.axvline(x=x_grid[index_start], color='darkblue', linestyle='--', linewidth=0.5,
                label='Граница области')
    plt.axvline(x=x_grid[index_end], color='darkblue', linestyle='--', linewidth=0.5)
    plt.title('Движение центра масс и коэффициент корреляции')
    plt.ylabel('y')
    plt.xlabel('t, ms')
    plt.legend()
    plt.savefig('centroid_norm.png', format='png')
    plt.show()


def find_angles_plot(c_mass, normalized_df, index_start, index_end):
    # Find angle of line
    n = len(c_mass)
    x_grid = np.linspace(interval[0], interval[0] + dt * n, n)
    xdelta = np.diff(x_grid)
    ydelta = np.diff(c_mass)
    res = np.rad2deg(np.arctan2(ydelta, xdelta))
    index = []
    for i in range(len(res)):
        if res[i] < angle_limit and i > index_start_dev:
            index.append(i)
    for point in index:
        plt.scatter(x_grid[point], normalized_df['y'][point], color='blue', marker='o', s=3)
    plt.plot(x_grid, correlations, linewidth=1, color='blue', label='Коэффициент корреляции')
    sns.lineplot(x_grid, normalized_df['y'], linewidth=1, color='deepskyblue', label='Траектория центра масс')
    plt.axvline(x=x_grid[index_start], color='darkblue', linestyle='--', linewidth=0.5,
                label='Граница области')
    plt.axvline(x=x_grid[index_end], color='darkblue', linestyle='--', linewidth=0.5)
    plt.title('Точки, подозрительные на движение по вертикали, общий график')
    plt.legend()
    plt.savefig('points_vertical.png', format='png')
    plt.show()
    print('Angle:', np.round(np.min(res), decimals=5))


def projection_plot_ts(x_grid, data_values, ind_start, ind_end, t_s, dt):
    time_start = x_grid[ind_start]
    time_end = x_grid[ind_end]
    ind_1 = int((time_start - t_s) // dt)
    ind_2 = int((time_end - t_s) // dt)
    fig, ax = plt.subplots(1, 3, figsize=(9, 4),
                        subplot_kw={'xticks': [2, 4, 6, 8, 10, 12, 14, 16],
                                    'yticks': [2, 4, 6, 8, 10, 12, 14, 16]})
    ax[0].imshow(data_values[:, :, ind_1], origin='lower')
    ax[0].set_title(f'{np.round(time_start, decimals=2)} ms')
    ax[1].imshow(data_values[:, :, ind_2], origin='lower')
    ax[1].set_title(f'{np.round(time_end, decimals=2)} ms')
    im = ax[2].imshow(np.fabs(np.subtract(data_values[:, :, ind_2], data_values[:, :, ind_1])), origin='lower')
    divider = make_axes_locatable(ax[2])
    cax = divider.append_axes('right', size='5%', pad=0.05)
    ax[2].set_title(f'Subtraction')
    fig.colorbar(im, cax=cax, orientation='vertical')
    fig.savefig("proj_diff.png", format="png")
    plt.show()


def save_images_for_video(data_values, t_s, dt):
    #time_start = x_grid[ind_start]   # for interval with loss of correlation
    #time_end = x_grid[ind_end]
    time_start = interval[0]   # for full interval
    time_end = interval[1]
    ind_1 = int((time_start - t_s) // dt)
    ind_2 = int((time_end - t_s) // dt)
    window_big = 10
    idx_current = ind_1
    i = 0
    n = (ind_2 - ind_1) // window_big
    max_value = np.max(data_values)
    min_value = np.min(data_values)
    print(max_value)
    print(min_value)
    while idx_current < ind_2:
        fig, ax = plt.subplots(subplot_kw={'xticks': [2, 4, 6, 8, 10, 12, 14, 16],
                                           'yticks': [2, 4, 6, 8, 10, 12, 14, 16]})
        data_renew = np.divide(np.subtract(data_values[:, :, idx_current], min_value), max_value - min_value)
        im = ax.imshow(data_renew, origin='lower')
        ax.set_title(f'{np.round(time_start + dt * i, decimals=3)} ms')
        divider = make_axes_locatable(ax)
        cax = divider.append_axes('right', size='5%', pad=0.05)
        plt.colorbar(im, cax=cax, orientation='vertical')
        file_name = 'pictures_full/proj_diff' + str(idx_current) + '.png'
        plt.savefig(file_name, format="png")
        #plt.show()
        plt.close(fig)
        idx_current += window_big
        i += window_big


if __name__ == '__main__':
    main_matrices, all_variables, indices = get_data_from_file()
    t, t_s, t_e, dt, B_up, B_low = prepare_extra_data(all_variables, main_matrices)

    sum_data = main_matrices.sum(axis=(0, 1))
    up_sum_data = B_up.sum(axis=(0, 1))
    low_sum_data = B_low.sum(axis=(0, 1))

    # Plot luminosity, whole and on interval
    # luminosity_plot(up_sum_data, low_sum_data, t, t_s)
    # # Correlation coefficient of south and north
    # correlations, x = correlation_coefficient(main_matrices)
    #
    # # Find borders of area with huge deviation
    # part_array = []
    # for i in range(len(correlations)):
    #     if correlations[i] <= 0.7:
    #         part_array.append(correlations[i])
    # index_start_dev = correlations.index(part_array[0])
    # index_end_dev = correlations.index(part_array[-1])
    #
    # # Centroid w/o normalization
    # c_mass = centroid(main_matrices)
    #
    # # Normalization
    # d = {'y': pd.Series(c_mass)}
    # df = pd.DataFrame(d)
    # normalized_df = (df - df.min()) / (df.max() - df.min())
    #
    # # Centroid and corrcoef
    # centroid_and_correlation(correlations, normalized_df, index_start_dev, index_end_dev)
    #
    # # Angles plot
    # find_angles_plot(c_mass, normalized_df, index_start_dev, index_end_dev)
    #
    # # Projection plot
    # projection_plot_ts(x, main_matrices, index_start_dev, index_end_dev, t_s, dt)
    save_images_for_video(main_matrices, t_s, dt)
    video.generate_video()


