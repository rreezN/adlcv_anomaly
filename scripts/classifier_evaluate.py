"""
Evaluate the classifier model on the test set.
"""
import argparse
from visdom import Visdom
viz = Visdom(port=8097)
import sys
sys.path.append("..")
sys.path.append(".")
from guided_diffusion.bratsloader import BRATSDataset
import numpy as np
import torch as th
from tqdm import tqdm

from torchvision.utils import save_image
from guided_diffusion.image_datasets import load_data
from guided_diffusion import dist_util, logger
from guided_diffusion.resample import create_named_schedule_sampler
from guided_diffusion.script_util import (
    classifier_and_diffusion_defaults,
    args_to_dict,
    add_dict_to_argparser,
    create_classifier_and_diffusion,

)

from classifier_sample_known import prepare_classifier_model


def main():
    args = create_argparser().parse_args()
    dist_util.setup_dist()
    logger.configure()

    logger.log("Setting up training set dataloader...")

    ds = BRATSDataset(args.data_dir, test_flag=True)
    dataload = th.utils.data.DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=False)
    
    
    model, diffusion = create_classifier_and_diffusion(
        **args_to_dict(args, classifier_and_diffusion_defaults().keys()),
    )
    
    logger.log("Preparing classifier model...")
    # Load the classifier model
    classifier_path = args.classifier_path
    classifier_model = prepare_classifier_model(args, classifier_path)

    if args.noised:
        schedule_sampler = create_named_schedule_sampler(
            args.schedule_sampler, diffusion, maxt=1000
        )


    # Create empty lists to store predictions and actual labels
    all_predictions = []
    all_actual_labels = []
    all_t_samples = []

    # Loop over images and evaluate accuracy using tqdm:
    for t_value in range(0, 1000, 50):
        datal = iter(dataload)
        for img in tqdm(datal):
            images = img[0].to(dist_util.dev())
            t =th.ones(len(images), dtype=th.long, device=dist_util.dev())*t_value
            print(f"t_value: {t_value}")
            
            if args.noised:
                # Sample t from schedule_sampler
                t, _ = schedule_sampler.sample(images.shape[0], dist_util.dev())
            
            images = diffusion.q_sample(images, t)
            
            with th.no_grad():
                pred = classifier_model(images, t)
                pred = th.argmax(pred, dim=1)
            
            # Assuming img[2] contains the actual labels
            actual = img[2]

            # print(pred)
            # print(actual)
            # print(t)

            # Append predictions and actual labels to lists
            all_predictions+=list(pred.cpu().numpy())
            all_actual_labels+=list(actual.cpu().numpy())
            all_t_samples+=list(t.cpu().numpy())

    # Convert lists to numpy arrays
    all_predictions = np.array(all_predictions)
    all_actual_labels = np.array(all_actual_labels)
    all_t_samples = np.array(all_t_samples)

    # Save predictions and actual labels as .npy files
    np.save('results/all_predictions.npy', all_predictions)
    np.save('results/all_actual_labels.npy', all_actual_labels)
    np.save('results/all_t_samples.npy', all_t_samples)

def create_argparser():
    defaults = dict(
        data_dir="data/brats21/processed/testing/",
        classifier_path="results/class/modelbratsclass095000.pt",
        batch_size=32,
        noised=True,
        schedule_sampler="uniform",
    )
    defaults.update(classifier_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser

if __name__ == "__main__":
    main()