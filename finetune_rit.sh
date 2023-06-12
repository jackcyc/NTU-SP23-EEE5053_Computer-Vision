# Reference: 
# https://github.com/AayushKrChaudhary/RITnet
# https://github.com/RSKothari/EllSeg


TRAINING_DATA_DIR=$1    # ../data/trainset
OUTPUT_DIR=$2           # output_dir

# go to the RITnet folder
cd ./RITnet/
# Download pre-train model provided by the author of EllSeg
wget -nc https://github.com/RSKothari/EllSeg/raw/master/weights/all.git_ok
# Main fine-tuning process
python finetune.py --expname $OUTPUT_DIR --data_path $TRAINING_DATA_DIR