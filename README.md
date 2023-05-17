# Unet

Full Unet implementation from original paper https://arxiv.org/pdf/1505.04597.pdf

## Architecture
In `unet_blocks.py` located `DoubleConv`, `Down` and `Up` blocks that make up the model. 

In `unet_model.py` located Unet model which consists of the above blocks.

<img width="1111" alt="image" src="https://github.com/EliseySoft/Unet/assets/81217562/8ffc6ad5-ac6e-481d-906b-4d5e470ddefb">


## Data
`PH2` dataset (https://www.fc.up.pt/addi/ph2%20database.html) was used to train the model. This image database contains a total of 200 dermoscopic images of melanocytic lesions, including 80 common nevi, 80 atypical nevi, and 40 melanomas. The PH² database includes medical annotation of all the images namely medical segmentation of the lesion, clinical and histological diagnosis and the assessment of several dermoscopic criteria (colors; pigment network; dots/globules; streaks; regression areas; blue-whitish veil).

## Training
Simple training code located in `train_unet.ipynb`. 
