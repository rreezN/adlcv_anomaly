import argparse
import sys
sys.path.append("..")
sys.path.append(".")
import os
import torch as th
import torch.nn as nn
import torch.distributions as td
from tqdm import tqdm
import random 
import numpy as np
from scipy.linalg import sqrtm
from scipy.stats import entropy

from guided_diffusion.bratsloader import BRATSDataset
from guided_diffusion.sampleloader import SampledDataset
from guided_diffusion import dist_util, logger
from guided_diffusion.script_util import (
    add_dict_to_argparser,
    args_to_dict,
    classifier_and_diffusion_defaults,
    create_classifier_and_diffusion,
)
from guided_diffusion.train_util import parse_resume_step_from_filename, log_loss_dict
from classifier_sample_known import prepare_classifier_model

def main():
    args = create_argparser().parse_args()
    dist_util.setup_dist()
    logger.configure()

    logger.log("Setting up training set dataloader...")

    ds = BRATSDataset(args.OG_data_dir)
    real_datal = th.utils.data.DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=False)
    
    logger.log("Setting up sampled set dataloader...")
    ds = SampledDataset(args.sampled_data_dir)
    fake_datal = th.utils.data.DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=False)

    logger.log("Beginning to compute FID...")
    frechet_inception_distance(args, real_datal, fake_datal)


def frechet_inception_distance(args, real_dataloader, fake_dataloader):
    """
    Compute the Frechet Inception Distance (FID) between real and fake samples.
    
    parameters:
    real_samples: [torch.Tensor]
        The real samples of dimension `(num_samples, num_features)`.
    fake_samples: [torch.Tensor]
        The fake samples of dimension `(num_samples, num_features)`.
    device: [torch.device]
        The device to use for the computation.
    returns:
    [float]
        The FID between the real and fake samples.
    """
    logger.log("Preparing classifier model...")
    # Load the classifier model
    classifier_path = args.classifier_path
    classifier_model = prepare_classifier_model(args, classifier_path)

    dims = 8192

    real_feat = np.empty((len(real_dataloader.dataset), dims))
    fake_feat = np.empty((len(fake_dataloader.dataset), dims))

    logger.log("Computing activations for real samples...")
    logger.log(f"Number of real images: {len(real_dataloader.dataset)}")

    # If file called "real_activations.npy" exists, load it and return
    if os.path.exists("data/brats21/processed/real_activations.npy"):
        real_feat = np.load("data/brats21/processed/real_activations.npy")
        logger.log("Loaded real activations from file.")
    
    else:
        start_idx = 0
        for images in tqdm(real_dataloader):
            img = images[0]
            img = img.to(dist_util.dev())
            t =th.zeros(len(img), dtype=th.long, device=dist_util.dev())
            real_activations = classifier_model.forward_feature_space(img, t).detach().cpu().numpy()
            # logger.log(f"real_activations.shape: {real_activations.shape}")
            activations_size = real_activations.shape[1]*real_activations.shape[2]*real_activations.shape[3]
            real_feat[start_idx:start_idx + real_activations.shape[0]] = real_activations.reshape(real_activations.shape[0],activations_size)
            start_idx = start_idx + real_activations.shape[0]
        np.save("data/brats21/processed/real_activations.npy", real_feat)

    # Compute the mean and covariance of the activations
    mu_real, sigma_real = real_feat.mean(axis=0), np.cov(real_feat, rowvar=False)
    logger.log(f"mu_real: {mu_real}")
    logger.log(f"sigma_real: {sigma_real}")

    logger.log("Computing activations for fake samples...")
    start_idx = 0
    logger.log(f"Number of fake samples: {len(fake_dataloader.dataset)}")
    for images in tqdm(fake_dataloader):
        img = images
        img = img.to(dist_util.dev())
        t =th.zeros(len(img), dtype=th.long, device=dist_util.dev())
        fake_activations = classifier_model.forward_feature_space(img,t).detach().cpu().numpy()
        activations_size = fake_activations.shape[1]*fake_activations.shape[2]*fake_activations.shape[3]
        fake_feat[start_idx:start_idx + fake_activations.shape[0]] = fake_activations.reshape(fake_activations.shape[0],activations_size)
        start_idx = start_idx + fake_activations.shape[0]

    # Compute the mean and covariance of the activations
    mu_fake, sigma_fake = fake_feat.mean(axis=0), np.cov(fake_feat, rowvar=False)
    logger.log(f"mu_fake: {mu_fake}")
    logger.log(f"sigma_fake: {sigma_fake}")

    logger.log("Computing FID...")
    # Compute the FID
    diff = mu_real - mu_fake
    covmean, _ = sqrtm(sigma_real.dot(sigma_fake), disp=False)
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    fid = diff.dot(diff) + np.trace(sigma_real) + np.trace(sigma_fake) - 2 * np.trace(covmean)
    logger.log(f"FID: {fid}")
    return fid

def create_argparser():
    defaults = dict(
        OG_data_dir="data/brats21/processed/training/training",
        sampled_data_dir="output_images/20240424-171020",
        classifier_path="results/class/modelbratsclass095000.pt",
        batch_size=32,
    )
    defaults.update(classifier_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser

if __name__ == "__main__":
    main()