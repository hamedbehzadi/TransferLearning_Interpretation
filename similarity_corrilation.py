from scipy.stats import pearsonr
import numpy as np
import glob
import matplotlib.pyplot as plt

from TransferLearningInterpretation.utils import create_if_not_exist, load_and_decompress_pickle

BASE_PATH = "/project_antwerp/thesis/"
SIM_OUTPUT_PATH = BASE_PATH + "similarities/"
GRAPH_OUTPUT_PATH = BASE_PATH + "graphs/"


class p_Data():
    def __init__(self) -> None:
        self.values = []
        self.labels = []


def run() -> None:
    sim_paths = glob.glob(SIM_OUTPUT_PATH + "**/LinCKA*.pbz2", recursive=True)
    sim_paths = sorted(sim_paths)

    correlations = []

    for sim_file in sorted(sim_paths):
        dataset, model, path = sim_file[:-5].split("/")[-3:]
        metric, _, net_epoch, feature = path.split("_")

        epoch = int(net_epoch[-2:])
        """
        Similarities file:
        (
            dict (path, class) -> ndarray(floats)
            dict (class) -> ndarray(floats)
        )
        """
        base_path = "/".join(sim_file[:-5].split("/")[:-1])
        lcka_similarities, lcka_avg_class_sim = load_and_decompress_pickle(f"{base_path}/LinCKA_pretrained_{net_epoch}_{feature}.pbz2")
        op_similarities, op_avg_class_sim = load_and_decompress_pickle(f"{base_path}/OrthPro_pretrained_{net_epoch}_{feature}.pbz2")

        for key in lcka_avg_class_sim:
            lcka = lcka_avg_class_sim[key]
            op = op_avg_class_sim[key]
            item_is = min(len(lcka), len(op))
            
            filter_mask = []
            for i in range(item_is):
                if np.isnan(lcka[i]) or np.isnan(op[i]):
                    filter_mask.append(False)
                else:
                    filter_mask.append(True)

            if sum(filter_mask) > 1:
                pc = pearsonr(lcka[filter_mask], op[filter_mask])
                print(path, key, pc)
                correlations.append(pc.statistic)

    c = np.array(correlations)
    print(np.min(c), np.max(c), np.mean(c), np.std(c))
        

if __name__ == "__main__":
    run()
