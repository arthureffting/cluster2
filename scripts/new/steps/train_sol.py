import argparse

parser = argparse.ArgumentParser(description='Prepare data for training')
parser.add_argument("--dataset", default="iam")
parser.add_argument("--base0", default=16)
parser.add_argument("--base1", default=16)
parser.add_argument("--alpha_alignment", default=0.1)
parser.add_argument("--alpha_backdrop", default=0.1)
parser.add_argument("--learning_rate", default=0.0001)
parser.add_argument("--crop_prob_label", default=0.5)
parser.add_argument("--crop_size", default=256)
parser.add_argument("--rescale_range", default=[384, 640])
parser.add_argument("--batch_size", default=1)
parser.add_argument("--images_per_epoch", default=1000)
parser.add_argument("--stop_after_no_improvement", default=20)
parser.add_argument("--max_epochs", default=1000)
parser.add_argument("--output", default="scripts/original/snapshots/training")
args = parser.parse_args()

print("[Training SOL finder] Model: SFRS | Dataset: ", args.dataset)

# TODO

print("[Not implemented]")
