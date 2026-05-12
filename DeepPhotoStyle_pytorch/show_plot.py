import argparse
import os
import re

import matplotlib.pyplot as plt


MODE_CONFIG = {
    "adv": {
        "pattern": r"adv loss: tensor\(\[?(\d+(?:\.\d+)?)\]?",
        "label": "adv loss",
        "title": "Adv Loss Over Steps",
        "ylabel": "Loss Value",
        "output_suffix": "_adv.png",
        "extract_message": "adv loss values",
    },
    "color": {
        "pattern": r"color_loss_: tensor\(\[?(\d+(?:\.\d+)?)\]?",
        "label": "adv loss",
        "title": "Color Loss Over Steps",
        "ylabel": "Loss Value",
        "output_suffix": "_color.png",
        "extract_message": "color loss values",
    },
    "con": {
        "pattern": r"Content Loss: (\d+\.\d+)",
        "label": "Content Loss",
        "title": "Content Loss Over Steps",
        "ylabel": "Loss Value",
        "output_suffix": "_content.png",
        "extract_message": "content loss values",
    },
    "content": {
        "pattern": r"content_loss: tensor\(\[?(\d+(?:\.\d+)?)\]?",
        "label": "adv loss",
        "title": "Content Loss Over Steps",
        "ylabel": "Loss Value",
        "output_suffix": "_content.png",
        "extract_message": "content loss values",
    },
    "midu": {
        "pattern": r"midu_loss: tensor\(\[?(\d+(?:\.\d+)?)\]?",
        "label": "adv loss",
        "title": "Midu Loss Over Steps",
        "ylabel": "Loss Value",
        "output_suffix": "_midu.png",
        "extract_message": "adv loss values",
    },
    "total": {
        "pattern": r"total loss: (\d+\.\d+)",
        "label": "Total Loss",
        "title": "Total Loss Over Steps",
        "ylabel": "Loss Value",
        "output_suffix": "_total.png",
        "extract_message": "total loss values",
    },
    "score": {
        "pattern": r"max_combined_score: tensor\((\d+\.\d+)",
        "label": "max_combined_score",
        "title": "Max Combined Score Over Steps",
        "ylabel": "Score",
        "output_suffix": ".png",
        "extract_message": "max_combined_score values",
        "ylim": (0.0, 1.0),
    },
}


def extract_values(file_path, pattern):
    with open(file_path, "r") as file:
        log_content = file.read()

    scores = re.findall(pattern, log_content)
    return [float(score) for score in scores]


def build_output_path(input_file, output_suffix):
    output_dir = os.path.dirname(input_file)
    file_name_without_ext = os.path.splitext(os.path.basename(input_file))[0]
    return os.path.join(output_dir, f"{file_name_without_ext}{output_suffix}")


def plot_and_save_scores(scores, output_path, config):
    plt.figure(figsize=(12, 6))
    plt.plot(scores, "b-", linewidth=2, label=config["label"])

    plt.title(config["title"])
    plt.xlabel("Step")
    plt.ylabel(config["ylabel"])
    plt.legend()
    plt.grid(True)

    if "ylim" in config:
        plt.ylim(*config["ylim"])

    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"Plot saved to {output_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description="Plot different metrics from a log file"
    )
    parser.add_argument(
        "--mode",
        required=True,
        choices=sorted(MODE_CONFIG.keys()),
        help="Choose which original show_*.py behavior to use",
    )
    parser.add_argument("input_file", help="Path to the input log file")
    args = parser.parse_args()

    config = MODE_CONFIG[args.mode]
    output_path = build_output_path(args.input_file, config["output_suffix"])
    scores = extract_values(args.input_file, config["pattern"])

    print(f"Extracted {len(scores)} {config['extract_message']}")
    plot_and_save_scores(scores, output_path, config)


if __name__ == "__main__":
    main()
