"""
Generate a large batch of image samples from a model and save them as a large
numpy array. This can be used to produce samples for FID evaluation.
"""
import matplotlib.pyplot as plt
import argparse
import os
from visdom import Visdom
viz = Visdom(port=8097)
import sys
sys.path.append("..")
sys.path.append(".")
from guided_diffusion.bratsloader import BRATSDataset
import torch.nn.functional as F
import numpy as np
import torch as th
import torch.distributed as dist
import torch
import torchvision
from datetime import datetime
from torchvision.utils import save_image
from guided_diffusion.image_datasets import load_data
from guided_diffusion import dist_util, logger
from guided_diffusion.script_util import (
    NUM_CLASSES,
    model_and_diffusion_defaults,
    classifier_defaults,
    create_classifier,
    create_model_and_diffusion,
    add_dict_to_argparser,
    args_to_dict,
)
def visualize(img):
    _min = img.min()
    _max = img.max()
    normalized_img = (img - _min)/ (_max - _min)
    return normalized_img

def prepare_model(args, model_path):
    model, diffusion = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys())
    )
    print("diff_model_path ", model_path)
    model.load_state_dict(
        dist_util.load_state_dict(model_path, map_location="cpu")
    )
    model.to(dist_util.dev())
    if args.use_fp16:
        model.convert_to_fp16()
    model.eval()
    return model, diffusion

def prepare_classifier_model(args, classifier_path):
    classifier = create_classifier(**args_to_dict(args, classifier_defaults().keys()))
    print("class_model_path ", classifier_path)

    classifier.load_state_dict(
        dist_util.load_state_dict(classifier_path)
    )
    classifier.to(dist_util.dev())
    if args.classifier_use_fp16:
        classifier.convert_to_fp16()
    classifier.eval()
    return classifier

def save_image_double(image, output_folder, file_name):
    # path1 = os.path.join(output_folder,"pngs", f'{file_name}.pdf')
    # save_image(image, path1)
    path1 = os.path.join(output_folder,"pngs", f'{file_name}.png')
    save_image(image, path1)
    path2 = os.path.join(output_folder,"numpys", f'{file_name}.npy')
    np.save(path2, image.to("cpu", torch.uint8).numpy())


def main():
    args = create_argparser().parse_args()

    # Create a folder to save images if it doesn't exist
    output_folder = os.path.join('output_images', f"noise{str(args.noise_level)}-s{args.classifier_scale}-" + datetime.now().strftime("%Y%m%d-%H%M%S"))
    os.makedirs(output_folder, exist_ok=True)
    os.makedirs(os.path.join(output_folder, "pngs"), exist_ok=True)
    os.makedirs(os.path.join(output_folder, "numpys"), exist_ok=True)

    dist_util.setup_dist()
    logger.configure()

    ds = BRATSDataset(args.data_dir, test_flag=True)
    datal = th.utils.data.DataLoader(
    ds,
    batch_size=args.batch_size,
    shuffle=False)

    datal = iter(datal)

    if args.ensemble:
        model_paths = [os.path.join(args.model_path, file) for file in os.listdir(args.model_path)]
        classifier_paths = [os.path.join(args.classifier_path, file) for file in os.listdir(args.classifier_path)]
    else:
        model_paths = [args.model_path]
        classifier_paths = [args.classifier_path]

    logger.log("sampling...")
    all_images = []
    all_labels = []

    imgNo = 0

    for img in datal:
        if img[2]!=0:
            imgNo += 1
            if imgNo > args.generate_number_images:
                break
            #Save input image:
            print('img no. ', imgNo,  img[0].shape, img[1])

            Labelmask = th.where(img[3] > 0, 1, 0)
            number=img[4][0]
            id = img[5][0]
            
            # Save img inputs
            for i in range(4):
                visualize_input = visualize(img[0][0, i, ...])
                save_image_double(visualize_input, output_folder, f'img_{id}_slice_{number}_input_{i}')

            # Save ground truth
            visualize_ground_truth = visualize(img[3][0, ...])
            save_image_double(visualize_ground_truth, output_folder, f'img_{id}_slice_{number}_ground_truth')

            if args.plot_banners:
                # # Combine input images and ground truth into one image
                resized_ground_truth = torchvision.transforms.functional.resize(img[3][0, ...], img[0].shape[-2:])

                combined_image = [img[0][0, i, ...] for i in range(4)] + [resized_ground_truth.unsqueeze(0)]

                # Visualize combined image with subplots
                fig, axes = plt.subplots(1, 5, figsize=(20, 4))
                for i in range(5):
                    #show the image from the combined image but reshape it to 2D:
                    axes[i].imshow(combined_image[i].reshape(256, 256), cmap='gray')
                    if i < 4:
                        axes[i].set_title(f'Input {i}')
                    else:
                        axes[i].set_title('Ground Truth')
                    axes[i].axis('off')

                # Save the combined image
                fig.savefig(os.path.join(output_folder, f'img_{id}_slice_{number}_input_and_ground_truth.pdf'))
                plt.close(fig)

            for k in range(len(model_paths)):
                model_path = model_paths[k]
                classifier_path = classifier_paths[k]
                print('model_path', model_path)
                print('classifier_path', classifier_path)

                logger.log("loading model and diffusion...")
                model, diffusion = prepare_model(args, model_path)
                logger.log("loaded model and diffusion")

                logger.log("loading classifier...")
                classifier = prepare_classifier_model(args, classifier_path)
                print('loaded classifier')

                def cond_fn(x, t,  y=None):
                    assert y is not None
                    with th.enable_grad():
                        x_in = x.detach().requires_grad_(True)
                        logits = classifier(x_in, t)
                        log_probs = F.log_softmax(logits, dim=-1)
                        selected = log_probs[range(len(logits)), y.view(-1)]
                        a=th.autograd.grad(selected.sum(), x_in)[0]
                        return  a, a * args.classifier_scale

                def model_fn(x, t, y=None):
                    assert y is not None
                    return model(x, t, y if args.class_cond else None)
                
                # p1 = np.array([np.array(p.shape).prod() for p in model.parameters()]).sum()
                # p2 = np.array([np.array(p.shape).prod() for p in classifier.parameters()]).sum()
                # print('pmodel', p1, 'pclass', p2)

                model_kwargs = {}

                if args.class_cond:
                    classes = th.randint(
                        low=0, high=1, size=(args.batch_size,), device=dist_util.dev()
                    )
                    model_kwargs["y"] = classes
                    print('y', model_kwargs["y"])
                sample_fn = (
                    diffusion.p_sample_loop_known if not args.use_ddim else diffusion.ddim_sample_loop_known
                )
                print('samplefn', sample_fn)
                start = th.cuda.Event(enable_timing=True)
                end = th.cuda.Event(enable_timing=True)
                start.record()
                sample, x_noisy, org = sample_fn(
                    model_fn,
                    (args.batch_size, 4, args.image_size, args.image_size), img, org=img,
                    clip_denoised=args.clip_denoised,
                    model_kwargs=model_kwargs,
                    cond_fn=cond_fn,
                    device=dist_util.dev(),
                    noise_level=args.noise_level,
                    output_folder=output_folder,
                    ensembleNo=k,
                    imgNo=id,
                    plot=args.plot_banners
                )
                end.record()
                th.cuda.synchronize()
                th.cuda.current_stream().synchronize()

                print('time for 1000', start.elapsed_time(end))

                # Save sampled outputs
                for i in range(4):
                    visualize_output = visualize(sample[0, i, ...])
                    save_image_double(visualize_output, output_folder, f'img_{id}_model_{k+1}_slice_{number}_sampled_output{i}')
                    


                # Calculate difference and save heatmap
                difftot = abs(org[0, :4, ...] - sample[0, ...]).sum(dim=0)
                visualize_difftot = visualize(difftot)
                save_image_double(visualize_difftot, output_folder, f'img_{id}_model_{k+1}_slice_{number}_difftot_heatmap')

                if args.plot_banners:
                    # Combine output images and heatmap into one image
                    combined_output = [visualize(sample[0, i, ...]) for i in range(4)] + [visualize(difftot)]

                    # Visualize combined output with subplots
                    fig, axes = plt.subplots(1, 5, figsize=(20, 4))
                    for i in range(5):
                        # show in grey tone:

                        if i < 4:
                            axes[i].imshow(torch.squeeze(combined_output[i],(0,1)).detach().cpu().numpy(), cmap='gray')
                            axes[i].set_title(f'Model {k+1} - Output {i}')
                        else:
                            axes[i].imshow(torch.squeeze(combined_output[i],(0,1)).detach().cpu().numpy())
                            axes[i].set_title(f'Model {k+1} -Heatmap')
                        axes[i].axis('off')

                    # Save the combined output
                    fig.savefig(os.path.join(output_folder, f'img_{id}_model_{k+1}_slice_{number}_output_and_heatmap.pdf'))
                    plt.close(fig)
        

            gathered_samples = [th.zeros_like(sample) for _ in range(dist.get_world_size())]
            dist.all_gather(gathered_samples, sample)  # gather not supported with NCCL
            all_images.extend([sample.cpu().numpy() for sample in gathered_samples])
            if args.class_cond:
                gathered_labels = [
                    th.zeros_like(classes) for _ in range(dist.get_world_size())
                ]
                dist.all_gather(gathered_labels, classes)
                all_labels.extend([labels.cpu().numpy() for labels in gathered_labels])
            

    arr = np.concatenate(all_images, axis=0)
    arr = arr[: args.num_samples]
    if args.class_cond:
        label_arr = np.concatenate(all_labels, axis=0)
        label_arr = label_arr[: args.num_samples]
    

    dist.barrier()
    logger.log("sampling complete")


def create_argparser():
    defaults = dict(
        data_dir="",
        clip_denoised=True,
        num_samples=10,
        batch_size=1,
        use_ddim=False,
        model_path="",
        classifier_path="",
        classifier_scale=100,
        noise_level=500,
        dataset='brats',
        ensemble=False,
        generate_number_images=1,
        plot_banners=True
    )
    defaults.update(model_and_diffusion_defaults())
    defaults.update(classifier_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser

if __name__ == "__main__":
    main()

