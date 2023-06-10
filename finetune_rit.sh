# https://github.com/AayushKrChaudhary/RITnet
# https://github.com/RSKothari/EllSeg

cd ./RITnet/
wget -nc https://github.com/RSKothari/EllSeg/raw/master/weights/all.git_ok

python finetune.py --expname finetune10ep_3e-4