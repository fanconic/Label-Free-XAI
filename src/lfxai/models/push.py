# Push prototypes to the nearest training image patches.
# Adapted from: https://github.com/cfchen-duke/ProtoPNet/blob/master/push.py

import torch
import numpy as np
import matplotlib.pyplot as plt
import cv2
import time
import os

from src.lfxai.utils.receptive_field import compute_rf_prototype
from src.lfxai.utils.helpers import makedir, find_high_activation_crop


# push each prototype to the nearest patch in the training set
def push_prototypes(
    dataloader,  # pytorch dataloader (must be unnormalized in [0,1])
    prototype_network,  # pytorch network with prototype_vectors
    pert,
    prototype_layer_stride=1,
    root_dir_for_saving_prototypes=None,  # if not None, prototypes will be saved here
    epoch_number=None,  # if not provided, prototypes saved previously will be overwritten
    prototype_img_filename_prefix=None,
    prototype_self_act_filename_prefix=None,
    proto_bound_boxes_filename_prefix=None,
    log=print,
    prototype_activation_function_in_numpy=None,
):
    prototype_network.eval()
    log("\tpush")

    start = time.time()
    prototype_shape = prototype_network.prototype_shape
    n_prototypes = prototype_network.n_prototypes
    # saves the closest distance seen so far
    global_min_proto_dist = np.full(n_prototypes, np.inf)
    # saves the patch representation that gives the current smallest distance
    global_min_fmap_patches = np.zeros(
        [n_prototypes, prototype_shape[1], prototype_shape[2], prototype_shape[3]]
    )

    """
    proto_rf_boxes and proto_bound_boxes column:
    0: image index in the entire dataset
    1: height start index
    2: height end index
    3: width start index
    4: width end index
    """

    proto_rf_boxes = np.full(shape=[n_prototypes, 5], fill_value=-1)
    proto_bound_boxes = np.full(shape=[n_prototypes, 5], fill_value=-1)

    if root_dir_for_saving_prototypes != None:
        if epoch_number != None:
            proto_epoch_dir = os.path.join(
                root_dir_for_saving_prototypes, "epoch-" + str(epoch_number)
            )
            makedir(proto_epoch_dir)
        else:
            proto_epoch_dir = root_dir_for_saving_prototypes
    else:
        proto_epoch_dir = None

    search_batch_size = dataloader.batch_size
    for push_iter, (search_batch_input, _) in enumerate(dataloader):
        """
        start_index_of_search keeps track of the index of the image
        assigned to serve as prototype
        """
        start_index_of_search_batch = push_iter * search_batch_size

        update_prototypes_on_batch(
            pert(search_batch_input),
            start_index_of_search_batch,
            prototype_network,
            global_min_proto_dist,
            global_min_fmap_patches,
            proto_rf_boxes,
            proto_bound_boxes,
            prototype_layer_stride=prototype_layer_stride,
            dir_for_saving_prototypes=proto_epoch_dir,
            prototype_img_filename_prefix=prototype_img_filename_prefix,
            prototype_self_act_filename_prefix=prototype_self_act_filename_prefix,
            prototype_activation_function_in_numpy=prototype_activation_function_in_numpy,
        )

    if proto_epoch_dir != None and proto_bound_boxes_filename_prefix != None:
        np.save(
            os.path.join(
                proto_epoch_dir,
                proto_bound_boxes_filename_prefix
                + "-receptive_field"
                + str(epoch_number)
                + ".npy",
            ),
            proto_rf_boxes,
        )
        np.save(
            os.path.join(
                proto_epoch_dir,
                proto_bound_boxes_filename_prefix + str(epoch_number) + ".npy",
            ),
            proto_bound_boxes,
        )

    log("\tExecuting push ...")
    prototype_update = np.reshape(global_min_fmap_patches, tuple(prototype_shape))
    prototype_network.protolayer.data.copy_(
        torch.tensor(prototype_update, dtype=torch.float32)
    )
    end = time.time()
    log("\tpush time: \t{0}".format(end - start))


# update each prototype for current search batch
def update_prototypes_on_batch(
    search_batch_input,
    start_index_of_search_batch,
    prototype_network,
    global_min_proto_dist,  # this will be updated
    global_min_fmap_patches,  # this will be updated
    proto_rf_boxes,  # this will be updated
    proto_bound_boxes,  # this will be updated
    prototype_layer_stride=1,
    dir_for_saving_prototypes=None,
    prototype_img_filename_prefix=None,
    prototype_self_act_filename_prefix=None,
    prototype_activation_function_in_numpy=None,
):
    prototype_network.eval()
    search_batch = search_batch_input

    with torch.no_grad():
        search_batch = search_batch
        protoL_input_torch, proto_dist_torch = prototype_network.push_forward(
            search_batch
        )

    protoL_input_ = np.copy(protoL_input_torch.cpu().numpy())
    proto_dist_ = np.copy(proto_dist_torch.cpu().numpy())
    del protoL_input_torch, proto_dist_torch

    prototype_shape = prototype_network.prototype_shape
    n_prototypes = prototype_shape[0]
    proto_h = prototype_shape[2]
    proto_w = prototype_shape[3]
    max_dist = prototype_shape[1] * prototype_shape[2] * prototype_shape[3]

    for j in range(n_prototypes):
        # search through every example
        proto_dist_j = proto_dist_[:, j, :, :]

        batch_min_proto_dist_j = np.amin(proto_dist_j)
        if batch_min_proto_dist_j < global_min_proto_dist[j]:
            batch_argmin_proto_dist_j = list(
                np.unravel_index(np.argmin(proto_dist_j, axis=None), proto_dist_j.shape)
            )
            # retrieve the corresponding feature map patch
            img_index_in_batch = batch_argmin_proto_dist_j[0]
            fmap_height_start_index = (
                batch_argmin_proto_dist_j[1] * prototype_layer_stride
            )
            fmap_height_end_index = fmap_height_start_index + proto_h
            fmap_width_start_index = (
                batch_argmin_proto_dist_j[2] * prototype_layer_stride
            )
            fmap_width_end_index = fmap_width_start_index + proto_w

            batch_min_fmap_patch_j = protoL_input_[
                img_index_in_batch,
                :,
                fmap_height_start_index:fmap_height_end_index,
                fmap_width_start_index:fmap_width_end_index,
            ]

            global_min_proto_dist[j] = batch_min_proto_dist_j
            global_min_fmap_patches[j] = batch_min_fmap_patch_j

            # get the receptive field boundary of the image patch that generates the representation
            protoL_rf_info = prototype_network.proto_layer_rf_info
            rf_prototype_j = compute_rf_prototype(
                search_batch.size(2), batch_argmin_proto_dist_j, protoL_rf_info
            )

            # get the whole image
            original_img_j = search_batch_input[rf_prototype_j[0]]
            original_img_j = original_img_j.cpu().numpy()
            original_img_j = np.transpose(original_img_j, (1, 2, 0))
            original_img_size = original_img_j.shape[0]

            # crop out the receptive field
            rf_img_j = original_img_j[
                rf_prototype_j[1] : rf_prototype_j[2],
                rf_prototype_j[3] : rf_prototype_j[4],
                :,
            ]

            # save the prototype receptive field information
            proto_rf_boxes[j, 0] = rf_prototype_j[0] + start_index_of_search_batch
            proto_rf_boxes[j, 1] = rf_prototype_j[1]
            proto_rf_boxes[j, 2] = rf_prototype_j[2]
            proto_rf_boxes[j, 3] = rf_prototype_j[3]
            proto_rf_boxes[j, 4] = rf_prototype_j[4]

            # find the highly activated region of the original image
            proto_dist_img_j = proto_dist_[img_index_in_batch, j, :, :]
            if prototype_network.prototype_activation_function == "log":
                proto_act_img_j = np.log(
                    (proto_dist_img_j + 1)
                    / (proto_dist_img_j + prototype_network.epsilon)
                )
            elif prototype_network.prototype_activation_function == "linear":
                proto_act_img_j = max_dist - proto_dist_img_j
            else:
                proto_act_img_j = prototype_activation_function_in_numpy(
                    proto_dist_img_j
                )
            upsampled_act_img_j = cv2.resize(
                proto_act_img_j,
                dsize=(original_img_size, original_img_size),
                interpolation=cv2.INTER_CUBIC,
            )
            proto_bound_j = find_high_activation_crop(upsampled_act_img_j)
            # crop out the image patch with high activation as prototype image
            proto_img_j = original_img_j[
                proto_bound_j[0] : proto_bound_j[1],
                proto_bound_j[2] : proto_bound_j[3],
                :,
            ]

            # save the prototype boundary (rectangular boundary of highly activated region)
            proto_bound_boxes[j, 0] = proto_rf_boxes[j, 0]
            proto_bound_boxes[j, 1] = proto_bound_j[0]
            proto_bound_boxes[j, 2] = proto_bound_j[1]
            proto_bound_boxes[j, 3] = proto_bound_j[2]
            proto_bound_boxes[j, 4] = proto_bound_j[3]

            if dir_for_saving_prototypes is not None:
                if prototype_self_act_filename_prefix is not None:
                    # save the numpy array of the prototype self activation
                    np.save(
                        os.path.join(
                            dir_for_saving_prototypes,
                            prototype_self_act_filename_prefix + str(j) + ".npy",
                        ),
                        proto_act_img_j,
                    )
                if prototype_img_filename_prefix is not None:
                    # save the whole image containing the prototype as png
                    if original_img_j.shape[2] == 1:
                        original_img_j = np.repeat(original_img_j, 3, axis=2)
                    if original_img_j.max() > 1 or original_img_j.min() < 0:
                        original_img_j = (original_img_j - original_img_j.min()) / (
                            original_img_j.max() - original_img_j.min()
                        )
                    plt.imsave(
                        os.path.join(
                            dir_for_saving_prototypes,
                            prototype_img_filename_prefix
                            + "-original"
                            + str(j)
                            + ".png",
                        ),
                        original_img_j,
                        vmin=0.0,
                        vmax=1.0,
                    )
                    # overlay (upsampled) self activation on original image and save the result
                    rescaled_act_img_j = upsampled_act_img_j - np.amin(
                        upsampled_act_img_j
                    )
                    rescaled_act_img_j = rescaled_act_img_j / np.amax(
                        rescaled_act_img_j
                    )
                    heatmap = cv2.applyColorMap(
                        np.uint8(255 * rescaled_act_img_j), cv2.COLORMAP_JET
                    )
                    heatmap = np.float32(heatmap) / 255
                    heatmap = heatmap[..., ::-1]
                    overlayed_original_img_j = 0.5 * original_img_j + 0.3 * heatmap
                    if overlayed_original_img_j.shape[2] == 1:
                        overlayed_original_img_j = np.repeat(
                            overlayed_original_img_j, 3, axis=2
                        )
                    if (
                        overlayed_original_img_j.max() > 1
                        or overlayed_original_img_j.min() < 0
                    ):
                        overlayed_original_img_j = (
                            overlayed_original_img_j - overlayed_original_img_j.min()
                        ) / (
                            overlayed_original_img_j.max()
                            - overlayed_original_img_j.min()
                        )
                    plt.imsave(
                        os.path.join(
                            dir_for_saving_prototypes,
                            prototype_img_filename_prefix
                            + "-original_with_self_act"
                            + str(j)
                            + ".png",
                        ),
                        overlayed_original_img_j,
                        vmin=0.0,
                        vmax=1.0,
                    )

                    # if different from the original (whole) image, save the prototype receptive field as png
                    if (
                        rf_img_j.shape[0] != original_img_size
                        or rf_img_j.shape[1] != original_img_size
                    ):
                        if rf_img_j.shape[2] == 1:
                            rf_img_j = np.repeat(rf_img_j, 3, axis=2)
                        if rf_img_j.max() > 1 or rf_img_j.min() < 0:
                            rf_img_j = (rf_img_j - rf_img_j.min()) / (
                                rf_img_j.max() - rf_img_j.min()
                            )
                        plt.imsave(
                            os.path.join(
                                dir_for_saving_prototypes,
                                prototype_img_filename_prefix
                                + "-receptive_field"
                                + str(j)
                                + ".png",
                            ),
                            rf_img_j,
                            vmin=0.0,
                            vmax=1.0,
                        )
                        overlayed_rf_img_j = overlayed_original_img_j[
                            rf_prototype_j[1] : rf_prototype_j[2],
                            rf_prototype_j[3] : rf_prototype_j[4],
                        ]
                        if overlayed_rf_img_j.shape[2] == 1:
                            overlayed_rf_img_j = np.repeat(
                                overlayed_rf_img_j, 3, axis=2
                            )
                        if overlayed_rf_img_j.max() > 1 or overlayed_rf_img_j.min() < 0:
                            overlayed_rf_img_j = (
                                overlayed_rf_img_j - overlayed_rf_img_j.min()
                            ) / (overlayed_rf_img_j.max() - overlayed_rf_img_j.min())
                        plt.imsave(
                            os.path.join(
                                dir_for_saving_prototypes,
                                prototype_img_filename_prefix
                                + "-receptive_field_with_self_act"
                                + str(j)
                                + ".png",
                            ),
                            overlayed_rf_img_j,
                            vmin=0.0,
                            vmax=1.0,
                        )

                    # save the prototype image (highly activated region of the whole image)
                    if proto_img_j.shape[2] == 1:
                        proto_img_j = np.repeat(proto_img_j, 3, axis=2)
                    if proto_img_j.max() > 1 or proto_img_j.min() < 0:
                        proto_img_j = (proto_img_j - proto_img_j.min()) / (
                            proto_img_j.max() - proto_img_j.min()
                        )
                    plt.imsave(
                        os.path.join(
                            dir_for_saving_prototypes,
                            prototype_img_filename_prefix + str(j) + ".png",
                        ),
                        proto_img_j,
                        vmin=0.0,
                        vmax=1.0,
                    )
