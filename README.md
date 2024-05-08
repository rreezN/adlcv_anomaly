# To Do list before Exam Deadline - 12th of May 2024

#### Code implementation
- [x] Transpose images to have the brain be vertically oriented as in the paper
- [x] Normalize images in the dataloader with torch transforms - between -1 and 1
#### To run on HPC
- [ ] Train the correct model with a cosine scheduler for the variance
#### Evaluations
- [ ] Classifier with accuracy and confusion matrix
- [ ] Threshold the anomaly images (OTSU???) and compared to GT segmentations (Segmentation scores or IoU or DICE). Can also be measured in accuracy, e.g. by saying if the IoU is under some threshold we count it as detected and otherwise NOPE.
- [ ] FID scores :P
- [ ] Assess the model with different L values - e.g. equidistantly run 10 different L values between some L less than the current L=500.
#### Poster
- [ ] Less text, preferebly bullet points
- [ ] Illustration of model architectures
- [ ] In Methods - have the training algorithm or showcase what happens during the training of the DDPM/DDIM ()
- [ ] In Methods - remove some of the main text
- [ ] Evaluate choice of font size
- [ ] Ensemble not necessary (nice to have) - If we wish to keep it, we need to calculate the anomaly maps for each model, and the take the absolute mean across the ensemble, before thresholding to create a binary segmentation mask. :)
- [ ] Investigate COO of the data. African data might be different. Difference between Brats21 and Brats20.

# Diffusion Models for Medical Anomaly Detection

We provide the Pytorch implementation of our MICCAI 2022 submission "Diffusion Models for Medical Anomaly Detection" (paper 704).


The implementation of Denoising Diffusion Probabilistic Models presented in the paper is based on [openai/guided-diffusion](https://github.com/openai/guided-diffusion).


## Data

We evaluated our method on the [BRATS2020 dataset](https://www.med.upenn.edu/cbica/brats2020/data.html), and on the [CheXpert dataset](https://stanfordmlgroup.github.io/competitions/chexpert/).
A mini-example how the data needs to be stored can be found in the folder *data*. To train or evaluate on the desired dataset, set `--dataset brats` or `--dataset chexpert` respectively. 

## Usage

We set the flags as follows:
```
MODEL_FLAGS="--image_size 256 --num_channels 128 --class_cond True --num_res_blocks 2 --num_heads 1 --learn_sigma True --use_scale_shift_norm False --attention_resolutions 16"
DIFFUSION_FLAGS="--diffusion_steps 1000 --noise_schedule linear --rescale_learned_sigmas False --rescale_timesteps False"
TRAIN_FLAGS="--lr 1e-4 --batch_size 10"
CLASSIFIER_FLAGS="--image_size 256 --classifier_attention_resolutions 32,16,8 --classifier_depth 4 --classifier_width 32 --classifier_pool attention --classifier_resblock_updown True --classifier_use_scale_shift_norm True"
SAMPLE_FLAGS="--batch_size 1 --num_samples 1 --timestep_respacing ddim1000 --use_ddim True"
```
To train the classification model, run
```
python scripts/classifier_train.py --data_dir path_to_traindata --dataset brats_or_chexpert $TRAIN_FLAGS $CLASSIFIER_FLAGS

Example:
python scripts/classifier_train.py --data_dir data/brats/processed/training/training  --dataset brats --lr 1e-4 --batch_size 10 --image_size 256 --classifier_attention_resolutions 32,16,8 --classifier_depth 4 --classifier_width 32 --classifier_pool attention --classifier_resblock_updown True --classifier_use_scale_shift_norm True   
                                                                                                                                      
```
To train the diffusion model, run
```
python scripts/image_train.py --data_dir --data_dir path_to_traindata --datasaet brats_or_chexpert  $MODEL_FLAGS $DIFFUSION_FLAGS $TRAIN_FLAGS

Example:
python scripts/image_train.py --data_dir  data/brats/processed/training/training --dataset brats --image_size 256 --num_channels 128 --class_cond True --num_res_blocks 2 --num_heads 1 --learn_sigma True --use_scale_shift_norm False --attention_resolutions 16 --diffusion_steps 1000 --noise_schedule linear --rescale_learned_sigmas False --rescale_timesteps False --lr 1e-4 --batch_size 10
```
The model will be saved in the *results* folder.

For image-to-image translation to a healthy subject on the test set, run
```
python scripts/classifier_sample_known.py  --data_dir path_to_testdata  --model_path ./results/model.pt --classifier_path ./results/classifier.pt --dataset brats_or_chexpert --classifier_scale 100 --noise_level 500 $MODEL_FLAGS $DIFFUSION_FLAGS $CLASSIFIER_FLAGS  $SAMPLE_FLAGS 
```
A visualization of the sampling process is done using [Visdom](https://github.com/fossasia/visdom).


## Comparing Methods

### FixedPoint-GAN

We follow the implementation given in this [repo](https://github.com/mahfuzmohammad/Fixed-Point-GAN). We choose λ<sub>cls</sub>=1, λ<sub>gp</sub>=λ<sub>id</sub>=λ<sub>rec</sub>=10,  and train our model for 150 epochs. The batch size is set to 10, and the learning rate to 10<sup>-4</sup>.

### VAE

We follow the implementation given in this [repo](https://github.com/aubreychen9012/cAAE) and train the model for 500 epochs.  The batch size is set to 10, and the learning rate to 10<sup>-4</sup>.


### DDPM
For sampling using the DDPM approach, run 
```
python scripts/classifier_sample_known.py  --data_dir path_to_testdata  --model_path ./results/model.pt --classifier_path ./results/classifier.pt  --dataset brats_or_chexpert --classifier_scale 100 --noise_level 500 $MODEL_FLAGS $DIFFUSION_FLAGS $CLASSIFIER_FLAGS 
```


