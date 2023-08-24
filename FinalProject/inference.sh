IMG_DIR=$1      # ./data/testing 
OUTPUT_DIR=$2   # ./

helpFunction()
{
   echo ""
   echo "Usage: bash $0  IMG_DIR  OUTPUT_DIR"
   echo -e "\tIMG_DIR:\tDirectory of the input images"
   echo -e "\tOUTPUT_DIR:\tDirectory of the outputs"
   exit 1 # Exit script after printing help
}

if [ -z "$IMG_DIR" ] || [ -z "$OUTPUT_DIR" ]
then
   echo "Some or all of the parameters are empty";
   helpFunction
fi

# inference segmentation part
python RITnet/inference.py --data_dir $IMG_DIR --output_dir $OUTPUT_DIR --ckpt './rit.pkl' 

# inference pupil occlusion part
python mae/inference.py --data_dir $IMG_DIR --output_dir $OUTPUT_DIR --ckpt_path './mae.pth'

# post processing
python post_processing.py --solution_path $OUTPUT_DIR
