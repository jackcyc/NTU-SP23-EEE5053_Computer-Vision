import glob
import matplotlib.pyplot as plt
import numpy as np
import os

def plot(root:str , set_name: str):

    # Find all .txt files under the root directory
    txt_files = sorted(glob.glob(root + '/**/*.txt', recursive=True))

    n_files = len(txt_files)

    # Define the number of subplots and the number of rows
    num_rows = (n_files + 7) // 8
    # Define the figure and axes for the subplots
    fig, axs = plt.subplots(num_rows, 8, figsize=(20, 2*num_rows))

    # Loop over the subplots and plot each one
    for i in range(n_files):
        # Compute the row and column indices for the subplot
        row_idx = i // 8
        col_idx = i % 8
        
        # Plot the data in the current subplot
        axs[row_idx, col_idx].set_ylim([-0.05, 1.05])
        axs[row_idx, col_idx].plot(np.loadtxt(txt_files[i]))
        
        # Add a title to the subplot
        axs[row_idx, col_idx].set_title(f'{set_name}-{i+1:02d}')

    # Adjust the spacing between the subplots
    fig.tight_layout()

    # # Save the plot to a file
    # plt.savefig(f'plot_{set_name}.png')
    fig.canvas.draw()
    img_buffer = np.array(fig.canvas.renderer.buffer_rgba())
    return img_buffer

# Define the root directory
root = '/mnt/191/a/ycc/CV_Final/RITnet/solution'
S_list = ['S5', 'S6', 'S7', 'S8'] 
img_buffers = []

for S in S_list:
    print(f'processing {S}')
    path2data = os.path.join(root, S)

    img_buffer = plot(path2data, S)
    img_buffers.append(img_buffer)

img = np.concatenate(img_buffers, axis=0)
plt.imsave(f'plot.png', img)


