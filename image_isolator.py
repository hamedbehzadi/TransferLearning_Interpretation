import glob
import os
import sys
import cv2
import numpy as np
from TransferLearningInterpretation.utils import create_if_not_exist
from TransferLearningInterpretation import config as cfg
# print(cv2.__version__)

VIS_OUTPUT_PATH = f"{cfg.BASE_PATH}visualisations/{cfg.MODEL_NAME}/"

def get_start(path: str) -> str:
    return "_".join(path.split("/")[-1].split("_")[:2])

class ImageIsolator(object):
    def __init__(self, basepath: str) -> None:
        t = basepath.split("/")
        self.model_name = t[-5]
        self.data_name = t[-4]
        self.layer_name = int(t[-3])
        self.basepath = basepath
        self.epoch_folders = ["pretrained", "epoch01", "epoch03", "epoch05", "epoch07", "epoch10", "epoch15"]
        self.epoch_names = ["Epoch 0", "Epoch 1", "Epoch 3", "Epoch 5", "Epoch 7", "Epoch 10", "Epoch 15"]
        self.output = basepath[:-7] + "vis/"
        create_if_not_exist(self.output)

        self.images = [get_start(i) for i in glob.glob(f"{basepath}/pretrained/*.png")]
    
    def get_images(self, image_prefix: str):
        images = []
        for epoch in self.epoch_folders:
            images_found = glob.glob(f"{self.basepath}/{epoch}/{image_prefix}*.png")

            if len(images_found) != 1:
                return []
            images.append(images_found[0])

        return images

    def create_images(self):
        for img in self.images:
            images = self.get_images(img)
            if len(images) == 0:
                continue

            W = 2000
            H = 400
            # [b, g, r, alpha]
            visualisation = np.full((H,W,3), 255, dtype=np.uint8)

            # # Calibri
            # ft = cv2.freetype.createFreeType2()
            # ft.loadFontData(fontFileName="/project_antwerp/thesis/calibri-font-family/calibri-regular.ttf", id=0)
            # ft.putText(img=visualisation,
            #     text='Quick Fox',
            #     org=(15, 70),
            #     fontHeight=60,
            #     color=(255,  255, 255),
            #     thickness=-1,
            #     line_type=cv2.LINE_AA,
            #     bottomLeftOrigin=True)

            font = cv2.FONT_HERSHEY_SIMPLEX
            font_size = 1.1
            font_color = (0,0,0)

            cl, fil = img.split("_")
            
            if self.model_name == "alexnet":
                m_name = "AlexNet"
            elif self.model_name == "vgg19":
                m_name = "VGG-19"
            elif self.model_name == "densenet":
                m_name = "DenseNet"
            elif self.model_name == "resnet50":
                m_name = "ResNet50"


            img_text = cv2.putText(visualisation, f"{m_name} {self.data_name}: Class {int(cl)}, Layer {self.layer_name}, Filter {int(fil)}", (20,40), font, 1.2, font_color, 2, cv2.LINE_AA)

            img_text = cv2.putText(visualisation, "LCKA:", (10,360), font, 0.8, font_color, 2, cv2.LINE_AA)
            img_text = cv2.putText(visualisation, "OP:", (10,390), font, 0.8, font_color, 2, cv2.LINE_AA)

            ix = 0
            for i in images:
                nx = 150 + ix*275
                ix += 1
                img_text = cv2.putText(visualisation, f"{self.epoch_names[ix-1]}", (nx,330), font, 0.8, font_color, 2, cv2.LINE_AA)
                sub_img = cv2.imread(i, 1)
                sub_img = cv2.resize(sub_img, (200,200), interpolation=cv2.INTER_AREA)
                h,w,_ = sub_img.shape
                nsx = nx - w//5
                nex = nsx + w

                visualisation[100:100+h,nsx:nex,:] = sub_img

                if ix > 1:
                    cka, opr = i[:-4].split("_")[-2:]

                    cka = float(cka)
                    opr = float(opr)

                    img_text = cv2.putText(visualisation, f"{cka*100:.2f}%", (nx,360), font, 0.8, font_color, 1, cv2.LINE_AA)
                    img_text = cv2.putText(visualisation, f"{opr*100:.2f}%", (nx,390), font, 0.8, font_color, 1, cv2.LINE_AA)


            cv2.imwrite(f"{self.output}{img}.png", visualisation)

            
def run() -> None:
    vis_dirs = glob.glob(VIS_OUTPUT_PATH + "**/epochs.tar.gz", recursive=True)

    for vis_dir in vis_dirs:
        print(f"Generating images for {vis_dir}")
        os.system(f"tar -xzf {vis_dir} --directory={vis_dir[:-13]} epochs/")
        c = ImageIsolator(f"{vis_dir[:-7]}/")
        c.create_images()
        os.system(f"rm -rf {vis_dir[:-7]}/")
        

if __name__ == "__main__":
    run()