import matplotlib.pyplot as plt
# from matplotlib import cm
import glob
import numpy as np
import cv2
import os
from TransferLearningInterpretation.utils import load_and_decompress_pickle, create_if_not_exist
from TransferLearningInterpretation import config as cfg


class HeatmapGenerator(object):
    def __init__(self, basepath: str) -> None:
        self.basepath = f"{cfg.BASE_PATH}heatmap/{cfg.DATASET_NAME}/{cfg.MODEL_NAME}/"
        self.activation_maps = sorted(glob.glob(f"{basepath}/*.pbz2"))
        self.epoch_names = ["Epoch 0", "Epoch 1", "Epoch 3",
                            "Epoch 5", "Epoch 7", "Epoch 10", "Epoch 15"]

    def generate_heatmaps(self) -> None:
        # Iterate over activation map dump files
        for a in self.activation_maps:
            data = load_and_decompress_pickle(a)

            epoch_id = a.split("_")[2][-2:]
            if epoch_id == "ed":
                epoch_id = "00"
            layer_ind = a.split("_")[-2][-3:]

            # Skip layers we visualised with other method (overlap useless + limit images generated to be reasonable)
            if int(layer_ind) not in cfg.HEATMAP_LAYER:
                continue

            print(a)

            # Iterate over input images in act. dump
            for j, k in enumerate(data.keys()):
                if "checkpoint" in k[0]:
                    continue

                if int(k[1]) != 2:
                    continue

                arr = data[k]
                original_image = cv2.imread(k[0], 1)
                h, w, _ = original_image.shape
                # original_image = cv2.resize(original_image, (224,224), interpolation=cv2.INTER_AREA)

                class_ind, img_ind = k[0][:-4].split("/")[-2:]

                for i in range(len(arr)):
                    create_if_not_exist(
                        f"{self.basepath}/{class_ind}/{img_ind}/")
                    d = arr[i]

                    d = cv2.resize(d, (w, h), interpolation=cv2.INTER_AREA)

                    # Create heatmap image
                    fig = plt.figure(frameon=False)
                    px = 1/plt.rcParams['figure.dpi']  # pixel in inches
                    fig.set_size_inches(224*px, 224*px)

                    ax = plt.Axes(fig, [0., 0., 1., 1.])
                    ax.set_axis_off()
                    fig.add_axes(ax)

                    ax.imshow(original_image, interpolation_stage="rgba")
                    ax.imshow(d, aspect="auto", cmap="jet",
                              alpha=0.75, interpolation_stage="rgba")
                    fig.savefig(
                        f"{self.basepath}{class_ind}/{img_ind}/{i}_{layer_ind}_{epoch_id}.png")
                    plt.close()

        return

    def generate_visualisations(self) -> None:
        images = sorted(glob.glob(f"{self.basepath}**/*.png", recursive=True))
        unique_images = {i[:-7] for i in images}

        for image in unique_images:
            W = 2000
            H = 400
            visualisation = np.full((H, W, 3), 255, dtype=np.uint8)

            font = cv2.FONT_HERSHEY_SIMPLEX
            font_size = 1.1
            font_color = (0, 0, 0)

            ft, layer = image.split("/")[-1].split("_")

            if cfg.MODEL_NAME == "alexnet":
                m_name = "AlexNet"
            elif cfg.MODEL_NAME == "vgg19":
                m_name = "VGG-19"
            elif cfg.MODEL_NAME == "densenet":
                m_name = "DenseNet"
            elif cfg.MODEL_NAME == "resnet50":
                m_name = "ResNet50"

            img_text = cv2.putText(
                visualisation, f"{m_name} {cfg.DATASET_NAME}: Layer {layer} Filter {ft}", (20, 40), font, 1.2, font_color, 2, cv2.LINE_AA)

            ix = 0
            for epoch in ["00", "01", "03", "05", "07", "10", "15"]:
                nx = 125 + ix*275
                ix += 1
                img_text = cv2.putText(
                    visualisation, f"{self.epoch_names[ix-1]}", (nx, 360), font, 0.8, font_color, 2, cv2.LINE_AA)
                sub_img = cv2.imread(f"{image}_{epoch}.png", 1)
                h, w, _ = sub_img.shape
                nsx = nx - w//5
                nex = nsx + w

                visualisation[100:100+h, nsx:nex, :] = sub_img
                os.remove(f"{image}_{epoch}.png")

            cv2.imwrite(f"{image}.png", visualisation)
        
        # os.system(f"tar -czf {self.basepath[:-1]}.tar.gz --directory={'/'.join(self.basepath.split('/')[:-2])}/ {self.basepath.split('/')[-2]}/")
        # os.system(f"rm -rf {self.basepath}")


if __name__ == "__main__":
    c = HeatmapGenerator(cfg.RF_OUTPUT_PATH)
    print("Generating individual images")
    c.generate_heatmaps()
    print("Generating grouped images")
    c.generate_visualisations()
