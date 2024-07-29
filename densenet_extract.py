from TransferLearningInterpretation import config as cfg
from TransferLearningInterpretation.model_extract import ModelExtract
from TransferLearningInterpretation.utils import check_need_calculate, save_and_compress_pickle


class DensenetExtract(ModelExtract):
    def _extract_layers(self, name) -> None:
        # denseblock 1
        output_path = f"{cfg.RF_OUTPUT_PATH}{name}_layer_block001_export.pbz2"
        if check_need_calculate(output_path, cfg.SKIP_EXISTING):
            l = self.model.features.denseblock1.denselayer6.relu2
            activations = self.rf_extract(l)
            save_and_compress_pickle(output_path, activations)

        # transition 1
        output_path = f"{cfg.RF_OUTPUT_PATH}{name}_layer_trans001_export.pbz2"
        if check_need_calculate(output_path, cfg.SKIP_EXISTING):
            l = self.model.features.transition1.relu
            activations = self.rf_extract(l)
            save_and_compress_pickle(output_path, activations)

        # denseblock 2
        output_path = f"{cfg.RF_OUTPUT_PATH}{name}_layer_block002_export.pbz2"
        if check_need_calculate(output_path, cfg.SKIP_EXISTING):
            l = self.model.features.denseblock2.denselayer12.relu2
            activations = self.rf_extract(l)
            save_and_compress_pickle(output_path, activations)

        # transition 2
        output_path = f"{cfg.RF_OUTPUT_PATH}{name}_layer_trans002_export.pbz2"
        if check_need_calculate(output_path, cfg.SKIP_EXISTING):
            l = self.model.features.transition2.relu
            activations = self.rf_extract(l)
            save_and_compress_pickle(output_path, activations)

        # denseblock 3
        output_path = f"{cfg.RF_OUTPUT_PATH}{name}_layer_block003_export.pbz2"
        if check_need_calculate(output_path, cfg.SKIP_EXISTING):
            l = self.model.features.denseblock3.denselayer24.relu2
            activations = self.rf_extract(l)
            save_and_compress_pickle(output_path, activations)

        # transition 3
        output_path = f"{cfg.RF_OUTPUT_PATH}{name}_layer_trans003_export.pbz2"
        if check_need_calculate(output_path, cfg.SKIP_EXISTING):
            l = self.model.features.transition3.relu
            activations = self.rf_extract(l)
            save_and_compress_pickle(output_path, activations)

        # denseblock 4
        output_path = f"{cfg.RF_OUTPUT_PATH}{name}_layer_block004_export.pbz2"
        if check_need_calculate(output_path, cfg.SKIP_EXISTING):
            l = self.model.features.denseblock4.denselayer16.relu2
            activations = self.rf_extract(l)
            save_and_compress_pickle(output_path, activations)
