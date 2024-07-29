import copy
import glob

import torch
import torch.nn as nn

from TransferLearningInterpretation.SimSiames.models.resnet import resnet50
from TransferLearningInterpretation.dataset import CustomActivationData
from TransferLearningInterpretation.train import load
from TransferLearningInterpretation.utils import (check_need_calculate, create_if_not_exist,
                                                  save_and_compress_pickle)

class ModelExtract():
    def __init__(self, dataset, cfg) -> None:
        self.data = dataset
        self.model = None
        self.MODEL_OUTPUT = []

    def model_hook(self,module, input_, output) -> None:
        self.MODEL_OUTPUT.append(output.squeeze(0).cpu().numpy())
    def rf_extract(self, layer, cfg):
        activations = dict()
        if isinstance(layer, torch.nn.ReLU):
            hook = layer.register_forward_hook(self.model_hook)
            self.model.eval()
            with torch.no_grad():
                for i, (inputs, _) in enumerate(self.data.test_loader):
                    _ = self.model(inputs.to(cfg.DEVICE))
                    activations[self.data.get_filename(i)] = self.MODEL_OUTPUT
                    self.MODEL_OUTPUT = []
            hook.remove()
        return activations

    def _extract_layers(self, name) -> None:
        for i in cfg.INSPECT_LAYERS_FEATURE:
            output_path = f"{cfg.RF_OUTPUT_PATH}{name}_layer_feature{i:03}_export.pbz2"
            if check_need_calculate(output_path, cfg.SKIP_EXISTING):
                l = self.model.features[i]
                activations = self.rf_extract(l)
                save_and_compress_pickle(output_path, activations)

        for i in cfg.INSPECT_LAYERS_CLASSIFIER:
            output_path = f"{cfg.RF_OUTPUT_PATH}{name}_layer_classifier{i:03}_export.pbz2"
            if check_need_calculate(output_path, cfg.SKIP_EXISTING):
                l = self.model.classifier[i]
                activations = self.rf_extract(l)
                save_and_compress_pickle(output_path, activations)

    def extract(self, cfg):
        create_if_not_exist(cfg.RF_OUTPUT_PATH)

        for model_path in glob.glob(cfg.MODEL_OUTPUT_PATH + "*.pth"):
            name = model_path.split('/')[-1][:-4]
            if cfg.MODEL_NAME == "resnet50":
                if 'pretrained' in model_path:
                    self.model = resnet50()
                else:
                    self.model = resnet50()
                    self.model.fc = nn.Linear(2048, cfg.NUM_CLASSES)
            load(self.model, model_path, cfg)
            self.model.eval()
            self.model.to(cfg.DEVICE)
            self._extract_layers(name, cfg)
