#!bin/bash
export PATH="/usr/local/cuda-9.0/bin:$PATH"
export LD_LIBRARY_PATH="/usr/local/cuda-9.0/lib64"
source ~/Documenti/tf-1.10/bin/activate

set -e
output_path=./output/
data_path_image=/media/filippo/nvme/ComputerVision/Dataset/FULL_KITTI/
filenames_file=./utils/filenames/kitti_2015_train.txt
temp=left_temp2

checkpoint_path=/home/filippo/Documenti/ECCV2020/MONO/ckpt/KITTI/bmvc/model-300000
width=1280
height=384
path_proxy=/media/filippo/nvme/ComputerVision/Dataset/FULL_KITTI/bmvc2k17_K/

output_folder=/media/filippo/Filippo/papers/ECCV2020/monocular_filtered_wild
nimages=25
mkdir -p $output_folder

python main.py --output_path $output_path \
				--data_path_image $data_path_image  \
				--data_path_proxy $path_proxy \
				--filenames_file $filenames_file \
				--checkpoint_path $checkpoint_path \
				--width $width \
				--height $height \
				--output_path $output_folder \
				--number_hypothesis $nimages \
				--temp_folder $temp \
				--right
