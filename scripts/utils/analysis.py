import os
import matplotlib.pyplot as plt

from scripts.utils.files import read_json


def store_plot(parameters):
    losses = {
        "training": [],
        "testing": [],
    }

    for epoch in range(1000):
        path = os.path.join(parameters.name, "epochs", str(epoch) + ".json")

        if os.path.exists(path):
            file_object = read_json(path)
            losses["training"].append(file_object["training_loss"])
            losses["testing"].append(file_object["testing_loss"])
        else:
            break

    v_errors = losses["testing"]
    t_errors = losses["training"]
    axes = plt.gca()
    axes.set_ylim([0, max(t_errors + v_errors)])
    axes.set_ylabel("Error")
    axes.set_xlabel("Epoch")
    plt.plot(t_errors, label="Testing", linestyle='--')
    plt.plot(v_errors, label="Validation")
    plt.legend()
    plot_path = os.path.join(parameters.name, "plot.png")
    plt.savefig(plot_path)
    plt.clf()
