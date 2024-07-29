import glob
from itertools import product

import numpy as np
import torch
# from multiprocess import Pool

from TransferLearningInterpretation import cka
from TransferLearningInterpretation.utils import (check_need_calculate, create_if_not_exist,
                                                  load_and_decompress_pickle, save_and_compress_pickle,
                                                  structural_similarity_index)

"""
Similarity file structure:
(
    dict (path, class) -> ndarray(floats)
    dict (class) -> ndarray(floats)
)
"""


def compute_similarities_PWSM(cfg, act_path):
    from InterpretationViaModelInversion.ICE.PWSM_metrics import LinearMetric
    """
        Computes the PWSM metric similarity for the provided files
    """

    cn1, cn2 = act_path
    n1, f1, p1 = cn1
    n2, f2, p2 = cn2

    # Checks if networks are not the same and features match
    if n1 == n2 or f1 != f2:
        return

    pwsm_file = f"{cfg.SIM_OUTPUT_PATH}PWSM_{n1}_{n2}_{f1}.pbz2"

    act1 = load_and_decompress_pickle(p1)
    act2 = load_and_decompress_pickle(p2)

    concatinated_acts1 = None
    concatinated_acts2 = None
    for batch_key in act1.keys():
        if concatinated_acts1 is None:
            concatinated_acts1 = [None for _ in range(len(act1[batch_key]))]
            concatinated_acts2 = [None for _ in range(len(act1[batch_key]))]
        for i in range(len(act1[batch_key])):
            temp_act1 = act1[batch_key][i]
            temp_act2 = act2[batch_key][i]
            if len(temp_act1.shape) == 3:
                temp_act1 = np.expand_dims(temp_act1,0)
                temp_act2 = np.expand_dims(temp_act2,0)
            if concatinated_acts1[i] is None:
                concatinated_acts1[i] = temp_act1
                concatinated_acts2[i] = temp_act2
            else:
                concatinated_acts1[i] = np.concatenate((concatinated_acts1[i],temp_act1),0)
                concatinated_acts2[i] = np.concatenate((concatinated_acts2[i],temp_act2),0)

    for i in range(len(concatinated_acts1)):
        concatinated_acts1[i] = np.swapaxes(concatinated_acts1[i],1,2)
        concatinated_acts1[i] = np.swapaxes(concatinated_acts1[i],2,3)
        concatinated_acts2[i] = np.swapaxes(concatinated_acts2[i],1,2)
        concatinated_acts2[i] = np.swapaxes(concatinated_acts2[i],2,3)

    copied_concatinated_acts1 = []
    copied_concatinated_acts2 = []
    for i in range(len(concatinated_acts1)):
        copied_concatinated_acts1.append(np.copy(concatinated_acts1[i]))
        copied_concatinated_acts2.append(np.copy(concatinated_acts2[i]))

        copied_concatinated_acts1[i] = np.mean(copied_concatinated_acts1[i],axis=(1,2))
        copied_concatinated_acts2[i] = np.mean(copied_concatinated_acts2[i],axis=(1,2))


    for i in range(len(concatinated_acts1)):
        temp_act1 = concatinated_acts1[i]
        concatinated_acts1[i] = temp_act1.reshape((temp_act1.shape[0] *
                                                   temp_act1.shape[1] *
                                                   temp_act1.shape[2],
                                                   temp_act1.shape[3]))
        temp_act2 = concatinated_acts2[i]
        concatinated_acts2[i] = temp_act2.reshape((temp_act2.shape[0] *
                                                   temp_act2.shape[1] *
                                                   temp_act2.shape[2],
                                                   temp_act2.shape[3]))


    pwsm_sim_unavgfilters = []
    pwsm_sim_avgfilters = []
    for i in range(len(concatinated_acts1)):
        metric = LinearMetric(alpha=0, center_columns=True, score_method="angular")
        metric.fit(concatinated_acts1[i], concatinated_acts2[i])
        dist = metric.score(concatinated_acts1[i], concatinated_acts2[i])
        pwsm_sim_unavgfilters.append(dist)

        metric = LinearMetric(alpha=0, center_columns=True, score_method="angular")
        metric.fit(copied_concatinated_acts1[i], copied_concatinated_acts2[i])
        dist = metric.score(copied_concatinated_acts2[i], copied_concatinated_acts2[i])
        pwsm_sim_avgfilters.append(dist)

    save_and_compress_pickle(pwsm_file, (pwsm_sim_unavgfilters, pwsm_sim_avgfilters))




def compute_similarities(cfg, act_path):
    """
    Computes the linear CKA and orthogonal procrustes similarity for the provided files
    """
    # orthp = OrthogonalProcrustes()
    lcka = cka.CudaCKA(cfg.DEVICE_CPU)

    cn1, cn2 = act_path
    n1, f1, p1 = cn1
    n2, f2, p2 = cn2

    # Checks if networks are not the same and features match
    if n1 == n2 or f1 != f2:
        return

    op_file = f"{cfg.SIM_OUTPUT_PATH}OrthPro_{n1}_{n2}_{f1}.pbz2"
    lc_file = f"{cfg.SIM_OUTPUT_PATH}LinCKA_{n1}_{n2}_{f1}.pbz2"
    ssim_file = f"{cfg.SIM_OUTPUT_PATH}ssim_{n1}_{n2}_{f1}.pbz2"

    run_op = check_need_calculate(op_file, cfg.SKIP_EXISTING)
    run_lc = check_need_calculate(lc_file, cfg.SKIP_EXISTING)
    run_ssim = check_need_calculate(ssim_file, cfg.SKIP_EXISTING)

    if run_op or run_lc or run_ssim:
        ac1 = load_and_decompress_pickle(p1)
        ac2 = load_and_decompress_pickle(p2)

        ac1_l = list(ac1.values())
        ac2_l = list(ac2.values())
        assert len(ac1_l) == len(ac2_l)

        ac1_shape = ac1_l[0].shape
        ac2_shape = ac2_l[0].shape
        assert ac1_shape == ac2_shape

        if len(ac1_shape) == 2:
            # Classifier
            op_similarities = dict()
            lc_similarities = dict()

            for key in ac1.keys():
                ci1 = torch.from_numpy(ac1[key])
                ci2 = torch.from_numpy(ac2[key])

                if run_op:
                    op_similarities[key] = float(
                        l2_sim_torch(ci1, ci2, sim_type='op_torch'))
                if run_lc:
                    lc_similarities[key] = float(lcka.linear_CKA(ci1, ci2))

            if run_op:
                save_and_compress_pickle(op_file, op_similarities)
            if run_lc:
                save_and_compress_pickle(lc_file, lc_similarities)

        elif len(ac1_shape) == 3:
            # Feature
            op_similarities = dict()
            lc_similarities = dict()
            ssim_similarities = dict()

            average_class_op_sim = dict()
            average_class_lc_sim = dict()

            for key in ac1.keys():
                _, class_id = key
                if class_id not in average_class_op_sim.keys():
                    average_class_op_sim[class_id] = []
                    average_class_lc_sim[class_id] = []

                ci1 = torch.from_numpy(ac1[key])
                ci2 = torch.from_numpy(ac2[key])

                # Amount of filters has to be the same for same layers
                assert ci1.shape[0] == ci2.shape[0]

                if run_op:
                    # Computing orthogonal Procrustes if needed
                    op_similarity = np.zeros(ci1.shape[0], dtype=np.float64)
                    for dim in range(ci1.shape[0]):
                        op_similarity[dim] = float(
                            l2_sim_torch(ci1[dim], ci2[dim], sim_type='op_torch'))
                    # op_similarities.append(np.nanmean(op_similarity))
                    op_similarities[key] = op_similarity
                    average_class_op_sim[class_id].append(op_similarity)
                if run_lc:
                    # Computing Linear CKA if needed
                    lc_similarity = np.zeros(ci1.shape[0], dtype=np.float64)
                    for dim in range(ci1.shape[0]):
                        x1 = normalize_matrix_for_similarity(ci1[dim], dim=1)
                        x2 = normalize_matrix_for_similarity(ci2[dim], dim=1)

                        lc_similarity[dim] = float(lcka.linear_CKA(x1, x2))
                    # lc_similarities.append(np.nanmean(lc_similarity))
                    lc_similarities[key] = lc_similarity
                    average_class_lc_sim[class_id].append(lc_similarity)
                if run_ssim:
                    # Computing ssim if needed
                    x1 = ci1.numpy()
                    x2 = ci2.numpy()
                    ssim_similarities[key] = structural_similarity_index(x1, x2)

            for class_id in average_class_op_sim.keys():
                average_class_op_sim[class_id] = np.nanmean(average_class_op_sim[class_id], axis=0)
                average_class_lc_sim[class_id] = np.nanmean(average_class_lc_sim[class_id], axis=0)

            if run_op:
                save_and_compress_pickle(op_file, (op_similarities, average_class_op_sim))
            if run_lc:
                save_and_compress_pickle(lc_file, (lc_similarities, average_class_lc_sim))
            if run_ssim:
                save_and_compress_pickle(ssim_file, ssim_similarities)


def run(cfg):
    create_if_not_exist(cfg.SIM_OUTPUT_PATH)

    # Collect all paths to receptive fields, extract model and features
    act_map_paths = [(it.split('/')[-1].split('_'), it)
                     for it in glob.glob(cfg.RF_OUTPUT_PATH + "*export.pbz2")]
    act_map_paths = [("".join(it[:-3]), it[-2], path)
                     for it, path in act_map_paths]
    # Create all possible combinations
    act_map_paths = list(product(act_map_paths, act_map_paths))

    # Filter useful combinations
    act_map_paths = list(
        filter(
            lambda cn: cn[0][0] == "pretrained" and cn[0][0] != cn[1][0] and
                       cn[0][1] == cn[1][1], act_map_paths))
    # For every combination, compute similarities
    for act_path in act_map_paths:
        # compute_similarities(cfg, act_path=act_path) # Linear CKA and Orthogonal Procrustic
        compute_similarities_PWSM(cfg, act_path)  # PWSM


