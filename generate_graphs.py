import glob
import csv

import matplotlib.pyplot as plt
import numpy as np

from TransferLearningInterpretation.utils import create_if_not_exist, load_and_decompress_pickle

BASE_PATH = "/project_antwerp/thesis/"
SIM_OUTPUT_PATH = BASE_PATH + "similarities/"
GRAPH_OUTPUT_PATH = BASE_PATH + "graphs/"


class p_Data():
    def __init__(self) -> None:
        self.values = []
        self.raw_values = []
        self.labels = []


def plot_sns_box_layer(plot_data, metric, model):

    if metric == "LinCKA":
        r_metric = "Linear CKA"
    elif metric == "OrthPro":
        r_metric = "Orthogonal Procrustes"

    if model == "alexnet":
        r_model = "AlexNet"
    elif model == "vgg19":
        r_model = "VGG-19"
    elif model == "densenet":
        r_model = "DenseNet"
    elif model == "resnet50":
        r_model = "ResNet50"

    markers = ["o", "v", "^", "<", ">", "s", "p", "D"]
    colors = ["dodgerblue", "orange", "r", "blueviolet", "lightgreen", "y"]

    for i, dataset in enumerate(plot_data):
        for j, feature in enumerate(plot_data[dataset]):
            plt.plot(plot_data[dataset][feature].labels, plot_data[dataset][feature].values,
                     marker=markers[j % len(markers)], color=colors[i % len(colors)], linestyle="--")

    plt.legend([f"{d} {e}" for d in plot_data.keys()
               for e in plot_data[d].keys()], bbox_to_anchor=(1.04, 1), loc="upper left")
    plt.title(f"{r_model}")
    plt.ylabel(f"Similarity({r_metric})")
    plt.xlabel("Epoch")
    plt.savefig(GRAPH_OUTPUT_PATH +
                f"{model}_{metric}.png", bbox_inches="tight")
    plt.close()

def plot_sns_box_filter(plot_data, metric, model):
    if metric == "LinCKA":
        r_metric = "Linear CKA"
    elif metric == "OrthPro":
        r_metric = "Orthogonal Procrustes"

    if model == "alexnet":
        r_model = "AlexNet"
    elif model == "vgg19":
        r_model = "VGG-19"
    elif model == "densenet":
        r_model = "DenseNet"
    elif model == "resnet50":
        r_model = "ResNet50"

    for i, dataset in enumerate(plot_data):
        for j, feature in enumerate(plot_data[dataset]):
            # base values 
            plt.figure(figsize=(19.2, 4.8))
            plt.set_cmap("Pastel1")
            for it, values in enumerate(plot_data[dataset][feature].raw_values):
                plt.plot(np.arange(0, len(values), 1), values, marker=".", linestyle="--")
            
            plt.legend([f"Epoch {s}" for s in plot_data[dataset][feature].labels], bbox_to_anchor=(1.04, 1), loc="upper left")
            plt.title(f"{r_model} {dataset} {feature}")
            plt.ylabel(f"Similarity({r_metric})")
            plt.xlabel("Filter")
            plt.xticks(np.arange(0, len(values), len(values)//20))
            
            plt.savefig(GRAPH_OUTPUT_PATH +
                        f"{model}_{metric}_test_{dataset}_{feature}.png", bbox_inches="tight")
            plt.close()

def plot_sns_box_layer_ssim(plot_data, metric, model):
    if model == "alexnet":
        r_model = "AlexNet"
    elif model == "vgg19":
        r_model = "VGG-19"
    elif model == "densenet":
        r_model = "DenseNet"
    elif model == "resnet50":
        r_model = "ResNet50"

    markers = ["o", "v", "^", "<", ">", "s", "p", "D"]
    colors = ["dodgerblue", "orange", "r", "blueviolet", "lightgreen", "y"]

    for i, dataset in enumerate(plot_data):
        for j, feature in enumerate(plot_data[dataset]):
            plt.plot(plot_data[dataset][feature].labels, np.nanmean(plot_data[dataset][feature].raw_values, axis=1).reshape((6,)),
                     marker=markers[j % len(markers)], color=colors[i % len(colors)], linestyle="--")

    plt.legend([f"{d} {e}" for d in plot_data.keys()
               for e in plot_data[d].keys()], bbox_to_anchor=(1.04, 1), loc="upper left")
    plt.title(f"{r_model}")
    plt.ylabel(f"Similarity(SSIM)")
    plt.xlabel("Epoch")
    plt.savefig(GRAPH_OUTPUT_PATH +
                f"{model}_ssim.png", bbox_inches="tight")
    plt.close()

def run() -> None:
    create_if_not_exist(GRAPH_OUTPUT_PATH)

    sim_paths = glob.glob(SIM_OUTPUT_PATH + "**/*.pbz2", recursive=True)
    plot_values = dict()

    for sim_file in sorted(sim_paths):
        dataset, model, path = sim_file[:-5].split("/")[-3:]
        metric, _, net_epoch, feature = path.split("_")

        if net_epoch == "bestloss":
            continue

        epoch = int(net_epoch[-2:])
        """
        Similarities file:
        (
            dict (path, class) -> ndarray(floats)
            dict (class) -> ndarray(floats)
        )
        """
        if metric != "ssim":
            similarities, avg_class_sim = load_and_decompress_pickle(sim_file)
        else:
            similarities = load_and_decompress_pickle(sim_file)

        if model not in plot_values:
            plot_values[model] = {metric: {dataset: {feature: p_Data()}}}
        elif metric not in plot_values[model]:
            plot_values[model][metric] = {dataset: {feature: p_Data()}}
        elif dataset not in plot_values[model][metric]:
            plot_values[model][metric][dataset] = {feature: p_Data()}
        elif feature not in plot_values[model][metric][dataset]:
            plot_values[model][metric][dataset][feature] = p_Data()

        plot_values[model][metric][dataset][feature].labels.append(f"{epoch}")
        if metric != "ssim":
            plot_values[model][metric][dataset][feature].values.append(np.nanmean(list(avg_class_sim.values())))
        plot_values[model][metric][dataset][feature].raw_values.append(np.nanmean(list(similarities.values()), axis=0))

    for model in plot_values:
        for metric in plot_values[model]:
            if metric != "ssim":
                plot_sns_box_layer(plot_values[model][metric],
                            metric=metric, model=model)
                plot_sns_box_filter(plot_values[model][metric],
                            metric=metric, model=model)
            else:

                plot_sns_box_layer_ssim(plot_values[model][metric],
                            metric=metric, model=model)
            

            # for dataset in plot_values[model][metric].keys():
            #     for feature in plot_values[model][metric][dataset].keys():
            #         with open(f"{GRAPH_OUTPUT_PATH}{model}_{metric}_test_{dataset}_{feature}.csv", "w", newline="") as csvfile:
            #             t = plot_values[model][metric][dataset][feature].raw_values
            #             labels = [f"{dataset} {feature} {i}" for i in range(len(t[0]))]

            #             csv_writer = csv.writer(csvfile, delimiter=',', dialect='excel')
            #             csv_writer.writerow(labels)
            #             csv_writer.writerows(t)


                    


if __name__ == "__main__":
    run()
