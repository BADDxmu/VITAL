import argparse

def parser():
    ap = argparse.ArgumentParser(description='data procseeing...')

    # For Prediction
    ap.add_argument('--load_list', type=str, help='Path to load list')
    ap.add_argument('--load_fasta', type=str, help='Path to load fasta folder')
    ap.add_argument('--save_path', type=str, help='Path to save features')

    args = ap.parse_args()
    return args
