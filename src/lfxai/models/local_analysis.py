# Functions for performing local_analysis.
# Used for running our experiments.


import torch
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import matplotlib.pyplot as plt
import cv2
import pandas as pd

import re
import os
import copy
from src.lfxai.models.proto_network import ProtoAutoEncoderMnist

from src.lfxai.utils.helpers import (
    makedir,
    find_high_activation_crop,
    visualize_image_grid,
)


def save_preprocessed_img(fname, preprocessed_imgs, index=0, save=False):
    img_copy = copy.deepcopy(preprocessed_imgs[index : index + 1].cpu().detach().numpy())
    undo_preprocessed_img = img_copy
    print("image index {0} in batch".format(index))
    undo_preprocessed_img = undo_preprocessed_img[0]
    undo_preprocessed_img = undo_preprocessed_img
    undo_preprocessed_img = np.transpose(undo_preprocessed_img, [1, 2, 0])

    if undo_preprocessed_img.max() > 1 or undo_preprocessed_img.min() < 0:
        undo_preprocessed_img = (
            undo_preprocessed_img - undo_preprocessed_img.min()
        ) / (undo_preprocessed_img.max() - undo_preprocessed_img.min())
    if save:
        if undo_preprocessed_img.shape[2] == 1:
            undo_preprocessed_img = np.repeat(undo_preprocessed_img, 3, axis=2)
        plt.imsave(fname, undo_preprocessed_img)
    return undo_preprocessed_img


def save_prototype(load_img_dir, fname, epoch, index, save=False):
    p_img = plt.imread(
        os.path.join(
            load_img_dir, "epoch-" + str(epoch), "prototype-img" + str(index) + ".png"
        )
    )
    if save:
        plt.axis("off")
        plt.imsave(fname, p_img)
    return p_img


def save_prototype_self_activation(load_img_dir, fname, epoch, index, save=False):
    p_img = plt.imread(
        os.path.join(
            load_img_dir,
            "epoch-" + str(epoch),
            "prototype-img-original_with_self_act" + str(index) + ".png",
        )
    )

    if save:
        plt.axis("off")
        plt.imsave(fname, p_img)
    return p_img


def save_prototype_original_img_with_bbox(
    load_img_dir,
    fname,
    epoch,
    index,
    bbox_height_start,
    bbox_height_end,
    bbox_width_start,
    bbox_width_end,
    color=(0, 255, 255),
    save=False,
):
    p_img_bgr = cv2.imread(
        os.path.join(
            load_img_dir,
            "epoch-" + str(epoch),
            "prototype-img-original" + str(index) + ".png",
        )
    )
    cv2.rectangle(
        p_img_bgr,
        (bbox_width_start, bbox_height_start),
        (bbox_width_end - 1, bbox_height_end - 1),
        color,
        thickness=1,
    )
    p_img_rgb = p_img_bgr[..., ::-1]
    p_img_rgb = np.float32(p_img_rgb) / 255

    if save:
        plt.imshow(p_img_rgb)
        plt.axis("off")
        plt.imsave(fname, p_img_rgb)
    return p_img_rgb


def imsave_with_bbox(
    fname,
    img_rgb,
    bbox_height_start,
    bbox_height_end,
    bbox_width_start,
    bbox_width_end,
    color=(0, 255, 255),
    save=False,
):
    img_bgr_uint8 = cv2.cvtColor(np.uint8(255 * img_rgb), cv2.COLOR_RGB2BGR)
    cv2.rectangle(
        img_bgr_uint8,
        (bbox_width_start, bbox_height_start),
        (bbox_width_end - 1, bbox_height_end - 1),
        color,
        thickness=1,
    )
    img_rgb_uint8 = img_bgr_uint8[..., ::-1]
    img_rgb_float = np.float32(img_rgb_uint8) / 255

    if save:
        plt.imshow(img_rgb_float)
        plt.axis("off")
        plt.imsave(fname, img_rgb_float)
    return img_rgb_float


class LocalAnalysis(object):
    def __init__(
        self,
        autoencoder: ProtoAutoEncoderMnist,
        prototypes_dir: str,
        epoch_number: int,
        image_save_directory: str,
        img_size: int = 32,
    ):
        """
        Perform local analysis.
        Arguments:
            load_model_dir (str): path to saved model directory.
            load_model_name (str): saved model name.
            test_image_name (str): test image file name.
            image_save_directory (str): directory to save images.
        """
        self.save_analysis_path = image_save_directory
        makedir(self.save_analysis_path)

        self.start_epoch_number = epoch_number

        self.ppnet = autoencoder

        self.img_size = img_size
        prototype_shape = self.ppnet.prototype_shape
        self.max_dist = prototype_shape[1] * prototype_shape[2] * prototype_shape[3]

        # confirm prototype class identity
        self.prototypes_dir = prototypes_dir
        self.prototype_info = np.load(
            os.path.join(
                self.prototypes_dir,
                "epoch-" + str(epoch_number),
                "bb" + str(epoch_number) + ".npy",
            )
        )

    def logclose(self):
        self.logclose()
        

    def visualization(
        self,
        input_image: torch.Tensor,
        img_name: str,
        target_image: torch.Tensor,
        show_images: bool = False,
        idx: int = 0,
        max_prototypes: int = 5,
        top_n: int = None,
    ):
        """
        Perform detailed local analysis comparing compressed and uncompressed images.
        Args:
            pil_img: is the PIL test image which is to be inspected
            img_name: name of the image to save it
            test_image_gt: ground truth reconstructed image
            show_images (default = False): Boolean value to show images
            idx (default = 0): Index Value, when retrieving results of the PPNet
            max_prototypes (int): number of most similar prototypes to display (default: 5)
            top_n (default = None): Visualize the top_n_th most activating prototype
        Returns:
            A list containing 5 images in the following order:
                1. Full picture of most activated prototype with bounding box
                2. Most activated prototype of image
                3. test sample, with activated patch in bounding box
                4. Corresponding activation map of the sample image
                5. Reconstructed image
                6. Ground truth image
        """
        # How to save the images
        specific_folder = self.save_analysis_path / img_name
        makedir(specific_folder)

        # Save activations
        dict_prototype_activations, dict_tables = {}, {}
        dict_prototype_activation_patterns = {}
        dict_array_act, dict_sorted_indices_act = {}, {}
        dict_original_img = {}

        # Forward the image variable through the network
        reconstruction, min_distances = self.ppnet(input_image)
        reconstruction = denormalize(reconstruction, (0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        conv_output, distances = self.ppnet.push_forward(input_image)
        prototype_activations = F.softmax(
            self.ppnet.distance_2_similarity(min_distances), dim=1
        )
        prototype_activation_patterns = self.ppnet.distance_2_similarity(distances)

        if self.ppnet.prototype_activation_function == "linear":
            prototype_activations = prototype_activations + self.max_dist
            prototype_activation_patterns = (
                prototype_activation_patterns + self.max_dist
            )

        print("------------------------------")

        original_img = save_preprocessed_img(
            os.path.join(specific_folder, f"original_img_{idx}.png"),
            input_image,
            idx,
            save=True,
        )
        reconstructed_img = save_preprocessed_img(
            os.path.join(specific_folder, f"reconstructed_img_{idx}.png"),
            reconstruction,
            idx,
            save=True,
        )
        target_img = save_preprocessed_img(
            os.path.join(specific_folder, f"target_img_{idx}.png"),
            target_image,
            idx,
            save=True,
        )

        ##### MOST ACTIVATED (NEAREST) PROTOTYPES OF THIS IMAGE
        makedir(os.path.join(specific_folder, "most_activated_prototypes"))
        dict_prototype_activations[0] = prototype_activations
        dict_prototype_activation_patterns[0] = prototype_activation_patterns
        dict_array_act[0], dict_sorted_indices_act[0] = torch.sort(
            prototype_activations[idx]
        )

        # Initialize as none, and will be filled after examining the first image
        inspected_index = None
        inspected_min = None
        inspected_max = None

        display_images = []
        similarity_values = []
        for i in range(1, max_prototypes + 1):
            if top_n is not None and top_n != i:
                continue

            if top_n is not None:
                if inspected_index is None:
                    inspected_index = dict_sorted_indices_act[0][-i].item()
                else:
                    i = np.where(
                        dict_sorted_indices_act[0].cpu().numpy() == inspected_index
                    )[0][0]
            else:
                inspected_index, inspected_min, inspected_max = None, None, None
                inspected_index = dict_sorted_indices_act[0][-i].item()

            similarity_values += [dict_array_act[0][-i]] * 7
            p_img = save_prototype(
                self.prototypes_dir,
                os.path.join(
                    self.save_analysis_path,
                    "most_activated_prototypes",
                    "top-%d_activated_prototype.png" % i,
                ),
                self.start_epoch_number,
                inspected_index,
            )

            p_oimg_with_bbox = save_prototype_original_img_with_bbox(
                self.prototypes_dir,
                fname=os.path.join(
                    self.save_analysis_path,
                    "most_activated_prototypes",
                    "top-%d_activated_prototype_in_original_pimg.png" % i,
                ),
                epoch=self.start_epoch_number,
                index=inspected_index,
                bbox_height_start=self.prototype_info[inspected_index][1],
                bbox_height_end=self.prototype_info[inspected_index][2],
                bbox_width_start=self.prototype_info[inspected_index][3],
                bbox_width_end=self.prototype_info[inspected_index][4],
                color=(0, 255, 255),
            )
            p_img_with_self_actn = save_prototype_self_activation(
                self.prototypes_dir,
                os.path.join(
                    self.save_analysis_path,
                    "most_activated_prototypes",
                    "top-%d_activated_prototype_self_act.png" % i,
                ),
                self.start_epoch_number,
                inspected_index,
            )

            activation_pattern = (
                dict_prototype_activation_patterns[0][idx][inspected_index]
                .cpu()
                .detach()
                .numpy()
            )
            upsampled_activation_pattern = cv2.resize(
                activation_pattern,
                dsize=(self.img_size, self.img_size),
                interpolation=cv2.INTER_CUBIC,
            )

            # show the most highly activated patch of the image by this prototype
            high_act_patch_indices = find_high_activation_crop(
                upsampled_activation_pattern
            )
            high_act_patch = original_img[
                high_act_patch_indices[0] : high_act_patch_indices[1],
                high_act_patch_indices[2] : high_act_patch_indices[3],
                :,
            ]

            plt.imsave(
                os.path.join(
                    specific_folder,
                    "most_activated_prototypes",
                    "most_highly_activated_patch_by_top-%d_prototype.png" % i,
                ),
                high_act_patch,
            )

            p_img_with_bbox = imsave_with_bbox(
                fname=os.path.join(
                    specific_folder,
                    "most_activated_prototypes",
                    "most_highly_activated_patch_in_original_img_by_top-%d_prototype.png"
                    % i,
                ),
                img_rgb=original_img,
                bbox_height_start=high_act_patch_indices[0],
                bbox_height_end=high_act_patch_indices[1],
                bbox_width_start=high_act_patch_indices[2],
                bbox_width_end=high_act_patch_indices[3],
                color=(0, 255, 255),
            )

            # show the image overlayed with prototype activation map and use normalization values of first run
            if inspected_min is None:
                inspected_min = np.amin(upsampled_activation_pattern)

            rescaled_activation_pattern = upsampled_activation_pattern - inspected_min

            if inspected_max is None:
                inspected_max = np.amax(rescaled_activation_pattern)

            rescaled_activation_pattern = rescaled_activation_pattern / inspected_max
            heatmap = cv2.applyColorMap(
                np.uint8(255 * rescaled_activation_pattern), cv2.COLORMAP_JET
            )
            heatmap = np.float32(heatmap) / 255
            heatmap = heatmap[..., ::-1]
            overlayed_img = 0.5 * original_img + 0.3 * heatmap

            p_img_test = original_img[
                high_act_patch_indices[0] : high_act_patch_indices[1],
                high_act_patch_indices[2] : high_act_patch_indices[3],
                :,
            ]

            plt.imsave(
                os.path.join(
                    specific_folder,
                    "most_activated_prototypes",
                    "prototype_activation_map_by_top-%d_prototype.png" % i,
                ),
                overlayed_img,
            )

            display_images += [
                p_oimg_with_bbox,
                p_img,
                p_img_with_bbox,
                overlayed_img,
                p_img_test,
                reconstructed_img,
                target_img,
            ]

            # Visualize Logs and Images

            display_titles = [
                "Training Image from which \nprototype is taken",
                "Prototype",
                "Test Image + BBox",
                "Test Image + Activation Map",
                "Test sample feature",
                "Reconstruction",
                "Ground Truth",
            ]

            visualize_image_grid(
                images=display_images,
                titles=display_titles,
                ncols=7,
                similarity_values=similarity_values,
                save_img=True,
                filename=specific_folder / "combination",
            )
            plt.tight_layout()
        if show_images:
            plt.show()

        return display_images


def denormalize(tensor, mean, std):
    """
    Denormalize a tensor image.

    Args:
        tensor (Tensor): Normalized tensor image of shape (C, H, W)
        mean (tuple): Mean values for each channel used for normalization
        std (tuple): Standard deviation values for each channel used for normalization

    Returns:
        Tensor: Denormalized image tensor
    """
    mean = torch.tensor(mean).view(3, 1, 1).to("cuda")
    std = torch.tensor(std).view(3, 1, 1).to("cuda")
    tensor = tensor * std + mean
    return tensor