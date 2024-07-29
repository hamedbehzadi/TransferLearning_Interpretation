import glob
import os

import cv2
import numpy as np

from TransferLearningInterpretation import config as cfg
from TransferLearningInterpretation.receptive_field import Receptive_Field
from TransferLearningInterpretation.utils import (check_need_calculate, create_if_not_exist,
                   load_and_decompress_pickle, check_if_exist)


RF_OUTPUT_PATH = f"{cfg.BASE_PATH}activation_maps/"
VIS_OUTPUT_PATH = f"{cfg.BASE_PATH}visualisations/{cfg.MODEL_NAME}/"


def compute_average_visualisation(act_path) -> None:
    RF = Receptive_Field(cfg.BASE_MODEL, None)
    dataset, model, epoch, layer, path = act_path

    layer_ind = int(layer[-3:])
    if layer_ind not in cfg.VISUALISATION_LAYER:
        return

    output_path = f"{VIS_OUTPUT_PATH}{dataset}/{layer_ind:03}/epochs/{epoch}/"
    print(output_path)

    if epoch != "pretrained":
        cka_sim = f"{cfg.BASE_PATH}similarities/{dataset}/{model}/LinCKA_pretrained_{epoch}_{layer}.pbz2"
        orp_sim = f"{cfg.BASE_PATH}similarities/{dataset}/{model}/OrthPro_pretrained_{epoch}_{layer}.pbz2"

        if not (check_if_exist(cka_sim) and check_if_exist(orp_sim)):
            print("ohno")
            return

        cka_sim = load_and_decompress_pickle(cka_sim)[1]
        orp_sim = load_and_decompress_pickle(orp_sim)[1]

    act_dump = load_and_decompress_pickle(path)

    for filter_ind in range(list(act_dump.values())[0].shape[0]):
        clipped_imgs = dict()
        weights = dict()
        sims = dict()
        for t, k in enumerate(act_dump.keys()):
            image_path, img_class = k

            if epoch != "pretrained":
                # N = 5
                # nan_cka = np.isnan(cka_sim[img_class]).sum()
                # nan_orp = np.isnan(orp_sim[img_class]).sum()

                # if nan_cka >= cka_sim[img_class].shape[0]-N:
                #     cka_filter_max_ind = []
                # else:
                #     cka_filter_max_ind = np.argpartition(cka_sim[img_class], -N-nan_cka)[-N-nan_cka:-nan_cka]
                # if nan_orp >= orp_sim[img_class].shape[0]-N:
                #     orp_filter_max_ind = []
                # else:
                #     orp_filter_max_ind = np.argpartition(orp_sim[img_class], -N-nan_orp)[-N-nan_orp:-nan_orp]
                # cka_filter_min_ind = np.argpartition(cka_sim[img_class], N)[:N]
                # orp_filter_min_ind = np.argpartition(orp_sim[img_class], N)[:N]

                img_sim_cka: float = cka_sim[img_class][filter_ind]
                img_sim_orp: float = orp_sim[img_class][filter_ind]
                # if (filter_ind in cka_filter_max_ind or filter_ind in orp_filter_max_ind) or (filter_ind in cka_filter_min_ind or filter_ind in orp_filter_min_ind):
                #     img_sim_cka:float = cka_sim[img_class][filter_ind]
                #     img_sim_orp:float = orp_sim[img_class][filter_ind]
                # else:
                #     continue
            else:
                img_sim_cka = 1.
                img_sim_orp = 1.

            if img_class not in clipped_imgs.keys():
                clipped_imgs[img_class] = []
                weights[img_class] = []
                sims[img_class] = []

            if not check_if_exist(image_path):
                continue

            activations = act_dump[k]
            a_filter = activations[filter_ind]

            # Check if the activation map contains only nan values
            if np.isnan(a_filter).sum() == a_filter.flatten().shape[0]:
                continue

            i = np.nanargmax(a_filter)
            area = RF.compute_rf_at_spatial_location(
                cfg.INPUT_SIZE[0], i // a_filter.shape[0], i % a_filter.shape[0], layer_ind)

            if area[1]-area[0] > 0 and area[3]-area[2] > 0:
                img = cv2.imread(image_path)
                img = cv2.resize(img, cfg.INPUT_SIZE,
                                 interpolation=cv2.INTER_AREA)
                # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                # img = cv2.copyMakeBorder(img, rf.padding, rf.padding, rf.padding, rf.padding, cv2.BORDER_CONSTANT, value=[255,255,255])

                clipped_img = img[area[0]:area[1], area[2]:area[3], :]
                if clipped_img.shape[0] == 0 or clipped_img.shape[1] == 0:
                    continue  # For some reason still can be empty sometimes
                clipped_imgs[img_class].append(clipped_img)
                # Collect max value, will act as weight for weighted average
                weights[img_class].append(np.nanmax(a_filter))
                sims[img_class].append(np.array((img_sim_cka, img_sim_orp)))

        for k in clipped_imgs.keys():
            if len(clipped_imgs[k]) > 0:
                target_shape = sorted(
                    [arr.shape for arr in clipped_imgs[k]], key=lambda shape: np.sum(shape))[0]

                if target_shape[0] < 32 or target_shape[1] < 32:
                    continue

                create_if_not_exist(output_path)

                reshaped_clipped_imgs = [cv2.resize(
                    src, (target_shape[0], target_shape[1]), interpolation=cv2.INTER_AREA) for src in clipped_imgs[k]]

                if np.sum(weights[k]) != 0:
                    average_img = np.average(
                        reshaped_clipped_imgs, weights=weights[k], axis=0)
                    average_img_sim = np.average(
                        sims[k], weights=weights[k], axis=0)
                    cv2.imwrite(
                        output_path + f"{k:02}_{filter_ind:03}_average_weighted_{average_img_sim[0]}_{average_img_sim[1]}.png", average_img)
                else:
                    average_img = np.average(reshaped_clipped_imgs, axis=0)
                    average_img_sim = np.average(sims[k], axis=0)
                    cv2.imwrite(
                        output_path + f"{k:02}_{filter_ind:03}_average_{average_img_sim[0]}_{average_img_sim[1]}.png", average_img)


def run() -> None:
    create_if_not_exist(VIS_OUTPUT_PATH)
    """
    act_maps: list[tuple]
        0: dataset
        1: model
        2: epoch
        3: layer
        4: full path
    """
    act_map_paths = [(path.split('/')[-3:], path)
                     for path in sorted(sorted(glob.glob(RF_OUTPUT_PATH + "**/*export.pbz2", recursive=True)))]

    act_map_paths = filter(
        lambda t: t[0][1] == cfg.MODEL_NAME and t[0][0] == cfg.DATASET_NAME, act_map_paths)

    act_map_paths = [(dataset, model, "".join(filename.split("_")[:-3]), filename.split("_")[-2], path)
                     for (dataset, model, filename), path in act_map_paths]

    for act_path in act_map_paths:
        compute_average_visualisation(act_path=act_path)

    for vis_dir in glob.glob(VIS_OUTPUT_PATH + "**/epochs/", recursive=True):
        print(f"Compressing images for {vis_dir}")
        os.system(f"tar -czf {vis_dir[:-1]}.tar.gz --directory={vis_dir[:-7]} epochs/")
        os.system(f"rm -rf {vis_dir}")


if __name__ == "__main__":
    run()
