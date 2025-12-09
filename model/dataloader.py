import numpy as np
import logging
import json
from typing import List, Dict, Any, Tuple

from torch.utils.data import Dataset, DataLoader
from model.utils import parser 

# Get global arguments (e.g., file paths, worker count)
args = parser()


# =======================================================================
# 1. Sequence Encoding Constant
# =======================================================================

CHARPROTSET = {
    "A": 1, "C": 2, "B": 3, "E": 4, "D": 5, "G": 6, "F": 7, "I": 8, "H": 9, 
    "K": 10, "M": 11, "L": 12, "O": 13, "N": 14, "Q": 15, "P": 16, "S": 17, 
    "R": 18, "U": 19, "T": 20, "W": 21, "V": 22, "Y": 23, "X": 0, "Z": 25,
}
CHARPROTLEN = 25 # Number of unique amino acids/symbols


# =======================================================================
# 2. Utility Functions for Padding and Normalization
# =======================================================================

def adjust_array(arr: np.ndarray, target_length: int) -> np.ndarray:
    """
    Adjusts the first dimension of a numpy array (sequence length) to the target length.
    
    If the current length is greater than target_length, the array is truncated.
    If the current length is less than target_length, the array is padded with zeros.

    Args:
        arr: Input numpy array, typically of shape (num_segments, feature_dim).
        target_length: The desired length for the first dimension.

    Returns:
        The adjusted numpy array, shaped (target_length, feature_dim).
    """
    num = arr.shape[0]
    
    if num > target_length:
        # Truncate
        return arr[:target_length]
    elif num < target_length:
        # Pad with zeros
        feature_dim = arr.shape[1]
        padding = np.zeros((target_length - num, feature_dim), dtype=arr.dtype)
        return np.vstack((arr, padding))
    else:
        # Length matches
        return arr

def make_matrix_center(lst: List[float], rows: int, cols: int, target_rows: int, target_cols: int) -> np.ndarray:
    """
    Pads a distance matrix (or interaction map) by placing the original matrix
    in the center of the target matrix, filling the rest with zeros.
    
    Args:
        lst: Flattened 1D list representing the original matrix (rows * cols elements).
        rows: Number of rows in the original matrix.
        cols: Number of columns in the original matrix.
        target_rows: Desired number of rows for the padded matrix.
        target_cols: Desired number of columns for the padded matrix.

    Returns:
        The padded numpy array, shaped (target_rows, target_cols).
    """
    # Create a target sized matrix initialized to zero
    matrix0 = np.zeros((target_rows, target_cols))

    # Convert 1D list back to 2D matrix (handling potential size mismatch with truncation)
    matrix = np.array([lst[i * cols:(i + 1) * cols] for i in range(rows)])

    A_rows, A_cols = matrix.shape

    # Calculate starting position for centering
    start_row = max(0, (target_rows - A_rows) // 2)
    start_col = max(0, (target_cols - A_cols) // 2)

    # Calculate effective area of A to be placed
    effective_rows = min(A_rows, target_rows)
    effective_cols = min(A_cols, target_cols)

    end_row = start_row + effective_rows
    end_col = start_col + effective_cols

    # Extract the effective part of the input matrix
    effective_matrix = matrix[:effective_rows, :effective_cols]

    # Place the effective matrix in the center of the target matrix
    matrix0[start_row:end_row, start_col:end_col] = effective_matrix

    return matrix0


def make_matrix_top_left(lst: List[float], rows: int, cols: int, target_rows: int, target_cols: int) -> np.ndarray:
    """
    Pads a distance matrix (or interaction map) by placing the original matrix
    in the top-left corner of the target matrix, filling the rest with zeros.
    
    Args:
        lst: Flattened 1D list representing the original matrix (rows * cols elements).
        rows: Number of rows in the original matrix.
        cols: Number of columns in the original matrix.
        target_rows: Desired number of rows for the padded matrix.
        target_cols: Desired number of columns for the padded matrix.

    Returns:
        The padded numpy array, shaped (target_rows, target_cols).
    """
    # Convert 1D list back to 2D matrix
    matrix = np.array([lst[i * cols:(i + 1) * cols] for i in range(rows)])

    # Handle truncation if original dimensions exceed target dimensions
    if rows > target_rows:
        matrix = matrix[:target_rows]
        rows = target_rows

    if cols > target_cols:
        matrix = matrix[:, :target_cols]
        cols = target_cols
    
    # Pad the matrix
    pad_width = ((0, target_rows - rows), (0, target_cols - cols))
    padded_array = np.pad(matrix, pad_width, 'constant', constant_values=(0, 0))
 
    return padded_array

def normalize_list(lst: List[float]) -> List[float]:
    """
    Min-Max normalization of a list of numbers.

    Args:
        lst: Input list of floats.

    Returns:
        Normalized list (values between 0 and 1).
    """
    min_val = min(lst)
    max_val = max(lst)
    
    if max_val == min_val:
        return [0.0] * len(lst) # Avoid division by zero
        
    normalized_lst = [(float(elem) - min_val) / (max_val - min_val) for elem in lst]
    return normalized_lst

def integer_label_protein(sequence: str, max_length: int = 15) -> np.ndarray:
    """
    Integer encoding (tokenization) for a protein string sequence based on CHARPROTSET.
    The sequence is truncated or padded implicitly to max_length.

    Args:
        sequence: Protein string sequence (e.g., 'ACGT...').
        max_length: Maximum encoding length of the input protein string.

    Returns:
        Numpy array of shape (max_length,) containing integer IDs.
    """
    encoding = np.zeros(max_length, dtype=np.int32)
    for idx, letter in enumerate(sequence[:max_length]):
        try:
            letter_upper = letter.upper()
            encoding[idx] = CHARPROTSET.get(letter_upper, 0) # Default to 0 ('X') if not found
        except Exception:
            logging.warning(
                f"Character {letter} does not exist in sequence category encoding, treating as padding."
            )
    return encoding


# =======================================================================
# 3. Custom Dataset and Collate Functions
# =======================================================================

class VITALDataset(Dataset):
    """
    Custom Dataset class for loading protein-peptide (or protein-protein) 
    interaction features for the VITAL model.
    """
    def __init__(self, pairs_idxs: List[str], labels: List[int], seq_x_max_length: int = 200, seq_y_max_length: int = 200):
        self.pairs_idxs = pairs_idxs
        self.labels = labels
        self.seq_x_max_length = seq_x_max_length
        self.seq_y_max_length = seq_y_max_length
    
    def __len__(self) -> int:
        return len(self.pairs_idxs)

    def __getitem__(self, index: int) -> Tuple:
        pair_idx = self.pairs_idxs[index]
        
        # Determine directory based on ID prefix (PePI or PPI)
        splited_id = pair_idx.split('_')
        data_dir = args.dir_feature_dict if splited_id[0] == 'PePI' else args.dir_ppi
        
        try:
            with open(data_dir + pair_idx, 'r') as f:
                features = json.load(f)
        except Exception as e:
            logging.error(f"Failed to load features for {pair_idx}: {e}")
            # In a real scenario, you might return placeholder data or skip
            raise e
        
        # --- Extract and Process Features ---
        
        # Segment-level features (numpy arrays)
        x_features = np.array(features['features_x'])
        y_features = np.array(features['features_y'])
        
        # Segment sequence strings
        x_segment_seq_str = features['chain_x']
        y_segment_seq_str = features['chain_y']

        # Segment interaction distance labels (for regression/segment-level prediction)
        segment_labels = features['seg_labels']

        # Global interaction label
        label = self.labels[index]

        # --- Process Segment Labels (Inverse Distance / Interaction Map) ---
        if label == 0: # Non-interacting pair
            # Set all segment-level interaction scores to 0.0
            segment_labels = [0.0 for _ in segment_labels]
        elif label == 1: # Interacting pair
            # Convert distance (d) into an inverse distance/interaction score (10.0/d)
            # This score represents the intensity of the interaction at the segment level.
            segment_labels = [10.0 / d if d != 0 else 0.0 for d in segment_labels]

        # Check for NaN (should not happen after the above logic, but good for debugging)
        if True in np.isnan(segment_labels):
            logging.warning(f"NaN found in segment labels for {pair_idx}")

        # Pad segment labels (interaction map) to max sequence lengths
        # Using make_matrix_top_left (renamed from original 'make_matrix')
        segment_labels_pad = make_matrix_top_left(
            segment_labels, 
            len(x_features), len(y_features),
            target_rows=self.seq_x_max_length, 
            target_cols=self.seq_y_max_length
        )

        # Pad segment-level features
        x_features_pad = adjust_array(x_features, target_length=self.seq_x_max_length)
        y_features_pad = adjust_array(y_features, target_length=self.seq_y_max_length)

        # Integer encode and pad segment sequences
        x_segment_seq_enc = adjust_array(
            np.array([integer_label_protein(seq, max_length=10) for seq in x_segment_seq_str]), 
            target_length=self.seq_x_max_length
        )
        y_segment_seq_enc = adjust_array(
            np.array([integer_label_protein(seq, max_length=10) for seq in y_segment_seq_str]), 
            target_length=self.seq_y_max_length
        )

        # Determine effective sequence lengths after potential truncation
        x_features_len = min(len(x_features), self.seq_x_max_length)
        y_features_len = min(len(y_features), self.seq_y_max_length)

        # Full sequence features (e.g., ESM-2 embeddings)
        # Note: Assuming full_seq_features_x/y are 1x640 arrays
        full_seq_features_x = np.asarray(features['esm2_feature_x']).reshape(1, 640)  
        full_seq_features_y = np.asarray(features['esm2_feature_y']).reshape(1, 640)

        return (
            x_features_pad, y_features_pad, segment_labels_pad, label, 
            x_features_len, y_features_len, len(segment_labels_pad), pair_idx, 
            x_segment_seq_enc, y_segment_seq_enc, 
            full_seq_features_x, full_seq_features_y
        )


class CustomCollate(object):
    """
    Collate function to handle batching of heterogeneous data (lists of numpy arrays/scalars).
    Since all arrays are pre-padded in VITALDataset, this primarily groups elements.
    """
    def __init__(self, device: str):
        self.device = device

    def collate_func(self, batch_list: List[Tuple]) -> Tuple:
        """
        Groups data samples into batches.

        Args:
            batch_list: A list of tuples, where each tuple is the output of __getitem__.

        Returns:
            A tuple containing lists of each data component in the batch.
        """
        # Unpack each component into separate lists
        x_features, y_features, \
        segment_labels, labels, \
        pep_segment_num, prot_segment_num, segment_label_num, \
        pair_idx, \
        x_segment_seq, y_segment_seq, \
        full_seq_features_x, full_seq_features_y = zip(*batch_list)

        # The data is returned as lists/tuples of numpy arrays, which will be converted 
        # to PyTorch tensors outside this function (e.g., in the training loop).
        return (
            list(x_features), list(y_features), list(segment_labels), list(labels),
            list(pep_segment_num), list(prot_segment_num), list(segment_label_num),
            list(pair_idx), list(x_segment_seq), list(y_segment_seq),
            list(full_seq_features_x), list(full_seq_features_y)
        )


def create_dataloader(
    dataset_idxs: List[str], 
    labels: List[int], 
    batch_size: int, 
    shuffle: bool, 
    seq_x_max_length: int, 
    seq_y_max_length: int
) -> DataLoader:
    """
    Initializes and returns a PyTorch DataLoader for the VITAL model.

    Args:
        dataset_idxs: List of IDs for the protein/peptide pairs.
        labels: List of global interaction labels (0 or 1).
        batch_size: Size of the batch.
        shuffle: Whether to shuffle the data (True for training, False for testing).
        seq_x_max_length: Maximum sequence length for the first chain (X).
        seq_y_max_length: Maximum sequence length for the second chain (Y).

    Returns:
        A PyTorch DataLoader instance.
    """
    dataset = VITALDataset(
        dataset_idxs, labels, 
        seq_x_max_length=seq_x_max_length, 
        seq_y_max_length=seq_y_max_length
    )  
    collate = CustomCollate(device=args.device)
    
    data_loader = DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=shuffle,
        num_workers=args.num_workers, 
        drop_last=True,
        collate_fn=collate.collate_func
    )
    return data_loader

# Alias the function used in train.py for backward compatibility in imports
# Note: The original train.py used 'dataloader_v5', so we'll provide the new name
# with a clear comment.
dataloader_v5 = create_dataloader