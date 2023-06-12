# Computer Vision Final Project

## Environment
* OS: Ubuntu 20.04
* GPU: NVIDIA GeForce RTX 3090 *2
* Create conda environment
    * `conda create --name <env> --file requirements.txt`

## Inference
```bash inference.sh <IMG_DIR> <OUTPUT_DIR>```
> The structure of IMG_DIR should be:
> ```
>    IMG_DIR
>    ├── S5
>    │   ├── 01
>    │   │   ├── 0.jpg
>    │   │   └── 1.jpg
>    │   │   └── ...
>    │   ├── 02
>    │   │   └── ...
>    │   └── ...
>    ├── S6
>    │   └── ...
>    ├── S7
>    │   └── ...
>    └── S8
>        └── ...
> ```
> Output will look like:
> ```
>    OUTPUT_DIR
>    └── solution
>        ├── S5
>        │   ├── 01
>        │   │   ├── 0.png
>        │   │   └── 1.png
>        │   │   └── ...
>        │   ├── 02
>        │   │   └── ...
>        │   └── ...
>        ├── S6
>        │   └── ...
>        ├── S7
>        │   └── ...
>        └── S8
>            └── ...
> ```

## Training
### Training data pre-processing
```shell
python data_preparation.py --data_source <TRAINING_DATA_DIR>
```
> ```
>    TRAINING_DATA_DIR
>    ├── S5
>    │   ├── 01
>    │   │   ├── 0.jpg
>    │   │   └── 1.jpg
>    │   │   └── ...
>    │   ├── 02
>    │   │   └── ...
>    │   └── ...
>    ├── S6
>    │   └── ...
>    ├── S7
>    │   └── ...
>    └── S8
>        └── ...
> ```

### Fine-tuning mae
```shell
bash finetune_mae.sh '../data/trainset' <OUTPUT_DIR>
# cp the trained model for inference
cp RITnet/<OUTPUT_DIR>/checkpoint-49.pth mae.pth
```
### Fine-tuning RITnet
```shell
bash finetune_rit.sh '../data/trainset' <OUTPUT_DIR>
# cp the trained model for inference
cp mae/<OUTPUT_DIR>/models/dense_net249.pkl rit.pkl
```


# Collaborators
* [crydaniel1203-bit](https://github.com/crydaniel1203-bit)
* [jackcyc](https://github.com/jackcyc)
* [tuchin32](https://github.com/tuchin32)