import argparse

parser = argparse.ArgumentParser(description='Generate character set for HTR')
parser.add_argument('--dataset')
args = parser.parse_args()

print("[Generating character set] Model: SFRS | Dataset: ", args.dataset)

# TODO
print("[Not implemented]")
