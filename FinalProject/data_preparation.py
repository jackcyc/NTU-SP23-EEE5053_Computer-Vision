import argparse
import os
import shutil
import zipfile
from glob import glob

import numpy as np
from PIL import Image
from tqdm import tqdm


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_source', '-s', type=str, required=True,
                        help='Path to the zip file or the directory of the training data')
    args = parser.parse_args()
    
    print('Arguments:')
    for arg in vars(args):
        print(f'    {arg}: {getattr(args, arg)}')

    return args

def build_training_data(src_path, s_list, dst_path):
    # Create directory
    os.makedirs(dst_path, exist_ok=True)

    for S in s_list:
        # Check if S is the training data
        if S not in ['S1', 'S2', 'S3', 'S4']:
            print(f'{S} is not the training data.')
            continue

        # Check if the zip file or the directory exists
        s_path = os.path.join(src_path, f'{S}')
        if not os.path.exists(f'{s_path}.zip') and not os.path.exists(s_path):
            print(f'Neither {s_path}.zip nor {s_path} exists.')
            continue

        # Extract zip file to the destination path, if the zip file exists
        # Otherwise, copy the directory to the destination path
        if os.path.exists(f'{s_path}.zip'):
            print(f'Extracting {S} ...')
            with zipfile.ZipFile(os.path.join(src_path, f'{S}.zip'), 'r') as zipf:
                zipf.extractall(dst_path)
        else:
            print(f'Copying {S} ...')
            try:
                shutil.copytree(s_path, os.path.join(dst_path, S))
            except OSError:
                shutil.rmtree(os.path.join(dst_path, S))
                shutil.copytree(s_path, os.path.join(dst_path, S))
        
        # Rename files and calculate labels
        d_dir = sorted(os.listdir(os.path.join(dst_path, S)))
        pbar = tqdm(d_dir)
        for d in pbar:
            pbar.set_description(f'Processing {S} {d}')
            file_list = sorted([name for name in os.listdir(os.path.join(dst_path, S, d)) if name.endswith('jpg') ],
                                key=lambda x: int(x.split('.')[0]))
            labels = []
            for file in file_list:
                number = int(file.split('.')[0])
                jpg_path = os.path.join(dst_path, S, d, file)
                png_path = jpg_path.replace('jpg', 'png')
                image = Image.open(png_path).convert('RGB')
                label = 0 if np.sum(image) == 0 else 1
                labels.append(label)
                os.rename(jpg_path, os.path.join(dst_path, S, d, f'{number:04d}.jpg'))
                os.rename(png_path, os.path.join(dst_path, S, d, f'{number:04d}.png'))
            np.save(os.path.join(dst_path, S, d, 'label.npy'), np.array(labels))

def train_val_split(dst_path):
    '''
    S1~S4 contains 19426 cases in total, 1669(8.6%) of which have label 0, and 17757(91.4%) have label 1.
    Randomly select a portion of subdirs to be val set.
    Results:
    # valdirs = ['S1/01', 'S1/11', 'S1/19', 'S2/05', 'S2/06', 'S2/20', 'S2/22', 'S3/02', 'S3/13', 'S3/14', 'S3/26', 'S4/11', 'S4/14', 'S4/16', 'S4/18']
    # len: 15
    # label 0: 343, label 1: 3728
    # ratio 0: 0.08425448292802751, ratio 1: 0.9157455170719725
    # train-val ratio: 0.14423076923076922
    # train-val ratio0: 343/1669=0.21, train-val ratio1: 3728/17757=0.21
    '''
    # Define the target number of cases with label 0 and 1
    np.random.seed(42)
    target_label_0 = (340, 390)
    target_label_1 = (3400, 3900)

    subdirs = sorted(glob(os.path.join(dst_path, '*', '*')))
    
    label = [np.load(os.path.join(dir, 'label.npy')) for dir in subdirs]
    label = np.concatenate(label, axis=0)
    casecnt_per_subdir = [len(sorted(glob(os.path.join(subdir, "*.jpg")))) for subdir in subdirs]
    labelcnt_per_subdir = [(np.sum(label[start:end] == 0), np.sum(label[start:end] == 1)) for start, end in zip(np.cumsum([0] + casecnt_per_subdir[:-1]), np.cumsum(casecnt_per_subdir))]

    # Select subdirectories until the target number of cases is reached

    selected_label_0 = 0
    selected_label_1 = 0
    while selected_label_0 < target_label_0[0] or selected_label_1 < target_label_1[0]:
        selected_subdirs = []
        casecnt_per_selected = []
        selected_label_0 = 0
        selected_label_1 = 0
        perm = np.random.permutation(len(subdirs))
        for idx in perm:
            subdir = subdirs[idx]
            casecnt = casecnt_per_subdir[idx]
            subdir_label_0 = labelcnt_per_subdir[idx][0]
            subdir_label_1 = labelcnt_per_subdir[idx][1]

            if selected_label_0 + subdir_label_0 <= target_label_0[1] and selected_label_1 + subdir_label_1 <= target_label_1[1]:
                selected_subdirs.append(subdir)
                casecnt_per_selected.append(casecnt)
                selected_label_0 += subdir_label_0
                selected_label_1 += subdir_label_1
            if selected_label_0 >= target_label_0[0] and selected_label_1 >= target_label_1[0]:
                break
        print(f'label 0: {selected_label_0}, label 1: {selected_label_1}')


    # Print the list of selected subdirectories
    res = sorted(selected_subdirs)
    print(res)
    print(f'len: {len(res)}')
    print(f'label 0: {selected_label_0}, label 1: {selected_label_1}')
    print(f'ratio 0: {selected_label_0 / (selected_label_0 + selected_label_1):.3f}, ratio 1: {selected_label_1 / (selected_label_0 + selected_label_1):.3f}')
    print(f'train-test ratio: {(selected_label_0 + selected_label_1) / sum(casecnt_per_subdir):.3f}')
    print(f'train-test dir ratio: {len(res) / len(subdirs):.3f}')

    # save the results to txt files
    metadata_train = sorted(list(set(subdirs) - set(res)))
    metadata_test = res

    with open(os.path.join(dst_path, 'metadata_train.txt'), 'w') as f:
        for item in metadata_train:
            f.write(f"{item[-5:]}\n")

    with open(os.path.join(dst_path, 'metadata_val.txt'), 'w') as f:
        for item in metadata_test:
            f.write(f"{item[-5:]}\n")
    
    with open(os.path.join(dst_path, 'metadata.txt'), 'w') as f:
        for item in sorted(subdirs):
            f.write(f"{item[-5:]}\n")


if __name__ == '__main__':
    
    args = get_args()
        
    # Build training data
    dst_path = './data/trainset'
    build_training_data(args.data_source, ['S1', 'S2', 'S3', 'S4'], dst_path)
    
    # generating metadata_train.txt and metadata_val.txt
    train_val_split(dst_path)