import numpy as np
from typing import Dict, Any, List, Tuple


# =======================================================================
# 1. Configuration and Constants
# =======================================================================

CHARPROTSET: Dict[str, int] = {
    "A": 1, "C": 2, "B": 3, "E": 4, "D": 5, "G": 6, "F": 7, "I": 8, "H": 9, 
    "K": 10, "M": 11, "L": 12, "O": 13, "N": 14, "Q": 15, "P": 16, "S": 17, 
    "R": 18, "U": 19, "T": 20, "W": 21, "V": 22, "Y": 23, "X": 0, "Z": 25,
}
CHARPROTLEN: int = 25 

DEFAULT_SEQ_SEGMENT_MAX_LEN: int = 100 # X/Y 特征的最大长度
DEFAULT_SEQ_ENCODING_MAX_LEN: int = 10  # 肽链/段序列整数编码的最大长度
DEFAULT_ESM2_FEATURE_DIM: int = 640     # ESM2 特征维度


# =======================================================================
# 2. Utility Functions
# =======================================================================

def adjust_array(arr: np.ndarray, target_length: int) -> np.ndarray:
    """
    Adjusts the first dimension of a NumPy array to the target length by 
    truncating or padding with zeros.

    The array is expected to have a shape of (N, D).

    Args:
        arr (np.ndarray): Input NumPy array, shape (N, D).
        target_length (int): The desired length for the first dimension.
    
    Returns:
        np.ndarray: The adjusted NumPy array, shape (target_length, D).
    """
    current_length = arr.shape[0]
    
    if current_length > target_length:
        # Truncate array
        return arr[:target_length]
    elif current_length < target_length:
        # Pad with zeros
        feature_dim = arr.shape[1]
        padding = np.zeros((target_length - current_length, feature_dim), dtype=arr.dtype)
        return np.vstack((arr, padding))
    else:
        # Length matches
        return arr

def make_matrix(lst: List[float], rows: int, cols: int, target_rows: int, target_cols: int) -> np.ndarray:
    """
    Converts a flat list of distance/label values into a 2D matrix (rows x cols) 
    and pads it with zeros to reach the target dimensions (target_rows x target_cols).
    The original matrix is placed at the top-left corner.

    Args:
        lst (List[float]): A flat list of values (e.g., distance matrix values).
        rows (int): The original number of rows.
        cols (int): The original number of columns.
        target_rows (int): The desired number of rows after padding.
        target_cols (int): The desired number of columns after padding.

    Returns:
        np.ndarray: The padded 2D array, shape (target_rows, target_cols).
    """
    matrix_2d: List[List[float]] = []
    
    # 1. Convert flat list to 2D matrix
    for i in range(rows):
        start = i * cols
        end = start + cols
        matrix_2d.append(lst[start:end])

    # 2. Truncation (if original dimensions exceed target)
    effective_rows = min(rows, target_rows)
    effective_cols = min(cols, target_cols)
    
    matrix_2d = matrix_2d[:effective_rows]
    matrix_2d = [row[:effective_cols] for row in matrix_2d]

    # 3. Padding
    matrix_np = np.array(matrix_2d)
    
    pad_width = (
        (0, target_rows - effective_rows),  # Pad below the matrix
        (0, target_cols - effective_cols)   # Pad to the right of the matrix
    )
    padded_array = np.pad(matrix_np, pad_width, 'constant', constant_values=0)
 
    return padded_array

def integer_label_protein(sequence: str, max_length: int = DEFAULT_SEQ_ENCODING_MAX_LEN) -> np.ndarray:
    """
    Performs integer encoding for a protein string sequence.

    Args:
        sequence (str): Protein string sequence.
        max_length (int): Maximum encoding length. Sequence longer than this will be truncated.
    
    Returns:
        np.ndarray: A 1D NumPy array of integers, shape (max_length,).
    """
    encoding = np.zeros(max_length, dtype=np.int32)
    
    for idx, letter in enumerate(sequence[:max_length]):
        letter = letter.upper()
        if letter in CHARPROTSET:
            encoding[idx] = CHARPROTSET[letter]
        else:
            # Optional: Log or print an issue for unknown characters
            # print(f"Warning: Character '{letter}' in sequence is unknown, treating as padding (0).")
            pass
            
    return encoding


# =======================================================================
# 3. Main Feature Processing
# =======================================================================

def process_feature_dict(
    features: Dict[str, Any], 
    seq_x_max_length: int = DEFAULT_SEQ_SEGMENT_MAX_LEN, 
    seq_y_max_length: int = DEFAULT_SEQ_SEGMENT_MAX_LEN
) -> Tuple[np.ndarray, np.ndarray, int, int, int, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Transforms a raw feature dictionary into padded NumPy arrays suitable for model input.

    Args:
        features (Dict[str, Any]): Dictionary containing raw features (e.g., 'features_x', 'chain_y', 'esm2_feature_x', etc.).
        seq_x_max_length (int): Maximum length for feature X (peptide) dimension.
        seq_y_max_length (int): Maximum length for feature Y (protein) dimension.

    Returns:
        Tuple: A tuple of 9 NumPy arrays representing the model inputs:
            1. x_features_pad (np.ndarray): Padded feature X. (seq_x_max_length, D)
            2. y_features_pad (np.ndarray): Padded feature Y. (seq_y_max_length, D)
            3. x_features_len (int): Effective length of feature X.
            4. y_features_len (int): Effective length of feature Y.
            5. segment_label_num (int): Number of segment labels.
            6. x_segment_seq (np.ndarray): Padded integer-encoded segment sequences X. 
            7. y_segment_seq (np.ndarray): Padded integer-encoded segment sequences Y.
            8. full_seq_features_x (np.ndarray): Global feature X (e.g., ESM2). (1, D_ESM2)
            9. full_seq_features_y (np.ndarray): Global feature Y (e.g., ESM2). (1, D_ESM2)
    """

    # --- 1. Raw Data Extraction and Initial Conversion ---
    # Convert lists to numpy arrays
    x_features_raw = np.array(features['features_x'])
    y_features_raw = np.array(features['features_y'])
    
    x_segment_seq_raw = features['chain_x']
    y_segment_seq_raw = features['chain_y']

    # Segment labels
    segment_labels_list = [0.0 for _ in features['seg_labels']]
    
    # --- 2. Length Calculation (Before Truncation/Padding) ---
    x_features_len_raw = x_features_raw.shape[0]
    y_features_len_raw = y_features_raw.shape[0]
    
    # Effective lengths for the model to use (min of raw length and max length)
    x_features_len = min(x_features_len_raw, seq_x_max_length)
    y_features_len = min(y_features_len_raw, seq_y_max_length)

    # --- 3. Padding and Adjustment ---

    # 3.1 Pad segment features (x_features, y_features)
    x_features_pad = adjust_array(x_features_raw, target_length=seq_x_max_length)
    y_features_pad = adjust_array(y_features_raw, target_length=seq_y_max_length)

    # 3.2 Process and pad segment sequence encoding (chain_x, chain_y)
    # First, encode each segment sequence using integer_label_protein
    x_encoded_segments = np.array([
        integer_label_protein(seq, max_length=DEFAULT_SEQ_ENCODING_MAX_LEN) 
        for seq in x_segment_seq_raw
    ])
    y_encoded_segments = np.array([
        integer_label_protein(seq, max_length=DEFAULT_SEQ_ENCODING_MAX_LEN) 
        for seq in y_segment_seq_raw
    ])
    
    # Then, adjust the resulting array of segment encodings
    x_segment_seq = adjust_array(x_encoded_segments, target_length=seq_x_max_length)
    y_segment_seq = adjust_array(y_encoded_segments, target_length=seq_y_max_length)

    # 3.3 Pad Segment Labels Matrix (Distance/Interaction Matrix)
    segment_labels_pad = make_matrix(
        lst=segment_labels_list, 
        rows=x_features_len_raw, 
        cols=y_features_len_raw,
        target_rows=seq_x_max_length, 
        target_cols=seq_y_max_length
    )
    segment_label_num = seq_x_max_length # The output dim of the matrix's first axis

    # 3.4 Process Global Features (ESM2)
    full_seq_features_x = np.asarray(features['esm2_feature_x']).reshape(1, DEFAULT_ESM2_FEATURE_DIM)
    full_seq_features_y = np.asarray(features['esm2_feature_y']).reshape(1, DEFAULT_ESM2_FEATURE_DIM)

    return (
        x_features_pad, 
        y_features_pad, 
        x_features_len,  # pep_segment_num 
        y_features_len,  # prot_segment_num 
        segment_label_num, 
        x_segment_seq, 
        y_segment_seq,
        full_seq_features_x,
        full_seq_features_y
    )