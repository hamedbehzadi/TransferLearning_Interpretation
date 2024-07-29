from TransferLearningInterpretation.model_extract import ModelExtract
from TransferLearningInterpretation.utils import check_need_calculate, save_and_compress_pickle


class Resnet50Extract(ModelExtract):
    def _extract_layers(self, name,cfg):
        ''''''
        '''Layer 0'''
        # relu layer
        output_path = f"{cfg.RF_OUTPUT_PATH}{name}_layer_feature000_export.pbz2"
        if check_need_calculate(output_path, cfg.SKIP_EXISTING):
            l = self.model.relu
            activations = self.rf_extract(l, cfg)
            save_and_compress_pickle(output_path, activations)
        '''Layer 1'''
        # Bottleneck 0
        output_path = f"{cfg.RF_OUTPUT_PATH}{name}_layer_feature010_export.pbz2"
        if check_need_calculate(output_path, cfg.SKIP_EXISTING):
            l = self.model.layer1[0].relu
            activations = self.rf_extract(l, cfg)
            save_and_compress_pickle(output_path, activations)
        # Bottleneck 1
        output_path = f"{cfg.RF_OUTPUT_PATH}{name}_layer_feature011_export.pbz2"
        if check_need_calculate(output_path, cfg.SKIP_EXISTING):
            l = self.model.layer1[1].relu
            activations = self.rf_extract(l, cfg)
            save_and_compress_pickle(output_path, activations)
        # Bottleneck 2
        output_path = f"{cfg.RF_OUTPUT_PATH}{name}_layer_feature012_export.pbz2"
        if check_need_calculate(output_path, cfg.SKIP_EXISTING):
            l = self.model.layer1[2].relu
            activations = self.rf_extract(l, cfg)
            save_and_compress_pickle(output_path, activations)
        '''Layer 2'''
        # Bottleneck 0
        output_path = f"{cfg.RF_OUTPUT_PATH}{name}_layer_feature020_export.pbz2"
        if check_need_calculate(output_path, cfg.SKIP_EXISTING):
            l = self.model.layer2[0].relu
            activations = self.rf_extract(l,cfg)
            save_and_compress_pickle(output_path, activations)
        # Bottleneck 1
        output_path = f"{cfg.RF_OUTPUT_PATH}{name}_layer_feature021_export.pbz2"
        if check_need_calculate(output_path, cfg.SKIP_EXISTING):
            l = self.model.layer2[1].relu
            activations = self.rf_extract(l, cfg)
            save_and_compress_pickle(output_path, activations)
        # Bottleneck 2
        output_path = f"{cfg.RF_OUTPUT_PATH}{name}_layer_feature022_export.pbz2"
        if check_need_calculate(output_path, cfg.SKIP_EXISTING):
            l = self.model.layer2[2].relu
            activations = self.rf_extract(l, cfg)
            save_and_compress_pickle(output_path, activations)
        # Bottleneck 3
        output_path = f"{cfg.RF_OUTPUT_PATH}{name}_layer_feature023_export.pbz2"
        if check_need_calculate(output_path, cfg.SKIP_EXISTING):
            l = self.model.layer2[3].relu
            activations = self.rf_extract(l, cfg)
            save_and_compress_pickle(output_path, activations)
        '''layer 3'''
        # Bottleneck 0
        output_path = f"{cfg.RF_OUTPUT_PATH}{name}_layer_feature030_export.pbz2"
        if check_need_calculate(output_path, cfg.SKIP_EXISTING):
            l = self.model.layer3[0].relu
            activations = self.rf_extract(l, cfg)
            save_and_compress_pickle(output_path, activations)
        # Bottleneck 1
        output_path = f"{cfg.RF_OUTPUT_PATH}{name}_layer_feature031_export.pbz2"
        if check_need_calculate(output_path, cfg.SKIP_EXISTING):
            l = self.model.layer3[1].relu
            activations = self.rf_extract(l, cfg)
            save_and_compress_pickle(output_path, activations)
        # Bottleneck 2
        output_path = f"{cfg.RF_OUTPUT_PATH}{name}_layer_feature032_export.pbz2"
        if check_need_calculate(output_path, cfg.SKIP_EXISTING):
            l = self.model.layer3[2].relu
            activations = self.rf_extract(l, cfg)
            save_and_compress_pickle(output_path, activations)
        # Bottleneck 3
        output_path = f"{cfg.RF_OUTPUT_PATH}{name}_layer_feature033_export.pbz2"
        if check_need_calculate(output_path, cfg.SKIP_EXISTING):
            l = self.model.layer3[3].relu
            activations = self.rf_extract(l, cfg)
            save_and_compress_pickle(output_path, activations)
        # Bottleneck 4
        output_path = f"{cfg.RF_OUTPUT_PATH}{name}_layer_feature034_export.pbz2"
        if check_need_calculate(output_path, cfg.SKIP_EXISTING):
            l = self.model.layer3[4].relu
            activations = self.rf_extract(l, cfg)
            save_and_compress_pickle(output_path, activations)
        # Bottleneck 5
        output_path = f"{cfg.RF_OUTPUT_PATH}{name}_layer_feature035_export.pbz2"
        if check_need_calculate(output_path, cfg.SKIP_EXISTING):
            l = self.model.layer3[5].relu
            activations = self.rf_extract(l, cfg)
            save_and_compress_pickle(output_path, activations)
        '''layer 4'''
        # Bottleneck 0
        output_path = f"{cfg.RF_OUTPUT_PATH}{name}_layer_feature040_export.pbz2"
        if check_need_calculate(output_path, cfg.SKIP_EXISTING):
            l = self.model.layer4[0].relu
            activations = self.rf_extract(l, cfg)
            save_and_compress_pickle(output_path, activations)
        # Bottleneck 1
        output_path = f"{cfg.RF_OUTPUT_PATH}{name}_layer_feature041_export.pbz2"
        if check_need_calculate(output_path, cfg.SKIP_EXISTING):
            l = self.model.layer4[1].relu
            activations = self.rf_extract(l, cfg)
            save_and_compress_pickle(output_path, activations)
        # Bottleneck 2
        output_path = f"{cfg.RF_OUTPUT_PATH}{name}_layer_feature042_export.pbz2"
        if check_need_calculate(output_path, cfg.SKIP_EXISTING):
            l = self.model.layer4[2].relu
            activations = self.rf_extract(l, cfg)
            save_and_compress_pickle(output_path, activations)
