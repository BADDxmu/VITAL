import os
import time
import json
import argparse
import datetime
import torch
from model.utils import parser
from parse_feature_dict import process_feature_dict
import traceback
import pandas as pd
import numpy as np

# =======================================================================
# 1. Configuration and Constants
# =======================================================================

# Prediction probability threshold for binary classification
PREDICTION_THRESHOLD = 0.5

# =======================================================================
# 2. Utility Functions for Model and Data
# =======================================================================

def load_checkpoint(filepath: str, device: str):
    """
    Loads a PyTorch model checkpoint and extracts the model.8

    Args:
        filepath (str): Path to the model checkpoint file.
        device (str): Computation device (e.g., 'cpu', 'cuda:0').

    Returns:
        torch.nn.Module: The loaded model in evaluation mode.
    """
    model_path = os.path.join(filepath, "VITAL.pt")

    # Load model
    model = torch.jit.load(model_path, map_location=device)
    model.eval()
    
    return model

def move_data_to_device(data, target_device, dtype=None):
    """
    Converts data to a PyTorch tensor and moves it to the specified device.

    Args:
        data: The input data (e.g., list, numpy array).
        target_device (str): The device to move the tensor to.
        dtype (torch.dtype, optional): Target data type. Defaults to None (uses torch.Tensor default).

    Returns:
        torch.Tensor: The tensor on the target device.
    """
    tensor = torch.tensor(data, dtype=dtype) if dtype else torch.Tensor(data)
    return tensor.to(target_device)


# =======================================================================
# 3. Prediction Logic
# =======================================================================

def get_prediction(feature_dict_path: str, model_ckpt: torch.nn.Module, device: str):
    """
    The main model inference logic: loads features, converts to tensors, and executes prediction.

    Args:
        feature_dict_path (str): Path to the feature dictionary JSON file.
        model_ckpt (torch.nn.Module): The loaded model instance.
        device (str): Computation device.

    Returns:
        tuple[int, float]: A tuple containing the predicted class (0 or 1) and the probability score.
    """
    if model_ckpt is None:
        raise RuntimeError("Model is not loaded. Cannot perform prediction.")

    # 1. Feature Preprocessing
    with open(feature_dict_path, 'r') as f:
        feature_dict = json.load(f)
        
    # Unpack features from the custom preprocessing function
    (x_features, y_features, pep_segment_num, prot_segment_num, 
     segment_label_num, x_segment_seq, y_segment_seq, 
     full_seq_features_x, full_seq_features_y) = process_feature_dict(feature_dict)

    # 2. Data Preparation and Device Movement
    x_features = move_data_to_device(x_features, device).unsqueeze(0)
    y_features = move_data_to_device(y_features, device).unsqueeze(0)
    full_seq_features_x = move_data_to_device(full_seq_features_x, device).unsqueeze(0)
    full_seq_features_y = move_data_to_device(full_seq_features_y, device).unsqueeze(0)
    x_segment_seq = move_data_to_device(x_segment_seq, device, torch.long).unsqueeze(0)
    y_segment_seq = move_data_to_device(y_segment_seq, device, torch.long).unsqueeze(0)
    pep_segment_num = torch.tensor(pep_segment_num, dtype=torch.long, device=device)
    prot_segment_num = torch.tensor(prot_segment_num, dtype=torch.long, device=device)
    segment_label_num = torch.tensor(segment_label_num, dtype=torch.long, device=device)

    # 3. Model Inference
    with torch.no_grad():
        probability, ASM = model_ckpt(
            x_features, y_features, pep_segment_num, prot_segment_num, 
            segment_label_num, x_segment_seq, y_segment_seq, 
            full_seq_features_x, full_seq_features_y
        )
    
    # Extract scalar probability value from the output tensor
    scalar_value = probability.cpu().item()
    predicted_class = 1 if scalar_value > PREDICTION_THRESHOLD else 0
    
    return predicted_class, scalar_value, ASM.cpu().numpy()


# =======================================================================
# 4. Command Line Interface
# =======================================================================

def save_asm_data(asm_array: np.ndarray, output_path: str, pair_id: str = None) -> str:
    """
    Save ASM numpy array to file with unique filename.
    
    Args:
        asm_array: ASM numpy array to save
        output_path: Directory or file path for saving
        pair_id: Optional pair ID for filename generation
        
    Returns:
        str: Path where ASM was saved
    """
    try:
        output_dir = os.path.dirname(output_path) if output_path.endswith('.npy') else output_path
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        
        # Generate unique filename
        if output_path.endswith('.npy'):
            asm_save_path = output_path
        else:
            if pair_id:
                filename = f"ASM_{pair_id}.npy"
            else:
                filename = f"ASM_output.npy"
            asm_save_path = os.path.join(output_path, filename)
        
        np.save(asm_save_path, asm_array)
        
        return asm_save_path
        
    except Exception as e:
        print(f"Failed to save ASM data: {e}")
        raise

def predict_single(args, model_ckpt) -> int:
    """
    Predict interaction for a single feature dictionary.
    
    Args:
        args: Command line arguments
        model_ckpt: Loaded model checkpoint
        
    Returns:
        Exit code (0 for success, 1 for error)
    """
    try:
        start_time = time.time()
        label, probability, ASM = get_prediction(args.feature_dict_path, model_ckpt, args.device)
        duration = time.time() - start_time
        
        # Save ASM data
        asm_save_path = save_asm_data(ASM, args.ASM_output_path)
        
        result = {
            "status": "success",
            "prediction": {
                "is_interaction": bool(label == 1),
                "probability": round(probability, 4),
                "predicted_label": int(label),
                "ASM_path": asm_save_path,
                "duration_seconds": round(duration, 2)
            },
            "metadata": {
                "input_file": os.path.basename(args.feature_dict_path),
                "model_checkpoint": os.path.basename(args.ckpt_path),
                "computation_device": args.device,
                "timestamp": time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time() + 8 * 3600))
            }
        }
        
        if args.verbose:
            print(f"\n Prediction completed in {duration:.2f} seconds")
            print(f"Result: Interaction={'YES' if label==1 else 'NO'}, Probability={probability:.4f}")
        
        output_dir = os.path.dirname(args.output)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        
        # Save to file
        with open(args.output, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=4, ensure_ascii=False)
        
        print(f"Prediction result saved to: {args.output}")
        return 0
        
    except Exception as e:
        print(f"Single prediction failed: {str(e)}")
        if args.verbose:
            traceback.print_exc()
        
        # Save error result
        error_result = {
            "status": "error",
            "error_message": str(e),
            "input_file": args.feature_dict_path,
            "timestamp": time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time() + 8 * 3600))
        }
        
        try:
            output_dir = os.path.dirname(args.output)
            if output_dir:
                os.makedirs(output_dir, exist_ok=True)
            
            with open(args.output, 'w', encoding='utf-8') as f:
                json.dump(error_result, f, indent=4, ensure_ascii=False)
            
            print(f"Error information saved to: {args.output}")
        except Exception as save_error:
            print(f"Failed to save error results: {save_error}")
        
        return 1

def predict_batch(args, model_ckpt) -> int:
    """
    Batch prediction for multiple feature dictionaries.
    
    Args:
        args: Command line arguments
        model_ckpt: Loaded model checkpoint
        
    Returns:
        Exit code (0 for success, 1 for error)
    """
    try:
        # Read input CSV
        if not os.path.exists(args.batch_input_csv):
            print(f"Batch input CSV not found: {args.batch_input_csv}")
            return 1
        
        df = pd.read_csv(args.batch_input_csv)
        
        # Validate required columns
        required_columns = ['pair_id', 'feature_dict_path']
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            print(f"Missing required columns in CSV: {missing_columns}")
            print(f"Required columns: {required_columns}")
            return 1
        
        if args.verbose:
            print(f"\nBatch prediction started")
            print(f"Input CSV: {args.batch_input_csv}")
            print(f"Total items: {len(df)}")
            print(f"Output file: {args.output}")
        
        results = []
        successful_predictions = 0
        failed_predictions = 0
        
        # Process each row
        for index, row in df.iterrows():
            pair_id = row['pair_id']
            feature_dict_path = row['feature_dict_path']
            
            if args.verbose:
                print(f"Processing: {pair_id} -> {feature_dict_path}")
            
            # Check if feature dictionary exists
            if not os.path.exists(feature_dict_path):
                print(f"Feature dictionary not found for {pair_id}: {feature_dict_path}")
                result = {
                    "pair_id": pair_id,
                    "status": "error",
                    "error_message": f"Feature dictionary not found: {feature_dict_path}",
                    "feature_dict_path": feature_dict_path
                }
                failed_predictions += 1
                results.append(result)
                continue
            
            try:
                # Perform prediction
                start_time = time.time()
                label, probability, ASM = get_prediction(feature_dict_path, model_ckpt, args.device)
                duration = time.time() - start_time

                # Save ASM data
                asm_save_path = save_asm_data(ASM, args.ASM_output_path, pair_id=pair_id)
                
                result = {
                    "pair_id": pair_id,
                    "prediction": {
                        "is_interaction": bool(label == 1),
                        "probability": round(probability, 4),
                        "predicted_label": int(label),
                        "ASM_path": asm_save_path,
                        "duration_seconds": round(duration, 2)
                    },
                    "feature_dict_path": feature_dict_path,
                    "timestamp": time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time() + 8 * 3600))
                }
                
                successful_predictions += 1
                if args.verbose:
                    print(f"{pair_id}: Interaction={'YES' if label==1 else 'NO'}, Prob={probability:.4f}")
                
            except Exception as e:
                print(f"Prediction failed for {pair_id}: {str(e)}")
                result = {
                    "pair_id": pair_id,
                    "status": "error",
                    "error_message": str(e),
                    "feature_dict_path": feature_dict_path,
                    "timestamp": time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time() + 8 * 3600))
                }
                failed_predictions += 1
            
            results.append(result)
        
        # Prepare final output
        batch_result = {
            "status": "completed",
            "summary": {
                "total_items": len(df),
                "successful_predictions": successful_predictions,
                "failed_predictions": failed_predictions,
                "success_rate": round(successful_predictions / len(df) * 100, 2) if len(df) > 0 else 0,
                "timestamp": time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time() + 8 * 3600))
            },
            "results": results
        }
        
        # Ensure output directory exists
        output_dir = os.path.dirname(args.output)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        
        # Save batch results
        with open(args.output, 'w', encoding='utf-8') as f:
            json.dump(batch_result, f, indent=4, ensure_ascii=False)
        
        # Print summary
        print(f"\nBatch prediction completed:")
        print(f"Successful: {successful_predictions}/{len(df)}")
        print(f"Failed: {failed_predictions}/{len(df)}")
        print(f"Success rate: {batch_result['summary']['success_rate']}%")
        print(f"Results saved to: {args.output}")
        
        return 0 if failed_predictions == 0 else 1
        
    except Exception as e:
        print(f"Batch prediction failed: {str(e)}")
        if args.verbose:
            traceback.print_exc()
        return 1

def main():
    """
    Main function for running the prediction script via command line.
    """    
    args = parser()
        
    if args.verbose:
        print(f"--- PePPI Prediction Initiated ---")
        print(f"Using device: {args.device}")
        if hasattr(args, 'feature_dict_path') and args.feature_dict_path:
            print(f"Single prediction input: {args.feature_dict_path}")
        if hasattr(args, 'batch_input_csv') and args.batch_input_csv:
            print(f"Batch prediction input: {args.batch_input_csv}")
        print(f"Output file: {args.output}")

    # 1. Model Loading
    model_ckpt = None
    try:
        model_ckpt = load_checkpoint(args.ckpt_path, args.device).to(args.device)
        print(f"Model loaded successfully on {args.device}")
    except Exception as e:
        print(f"Fatal Error: Failed to load model from {args.ckpt_path}. Details: {e}")
        return 1

    # 2. Prediction Execution
    try:
        if hasattr(args, 'batch_input_csv') and args.batch_input_csv:
            # Batch prediction mode
            return predict_batch(args, model_ckpt)
        else:
            # Single prediction mode
            return predict_single(args, model_ckpt)
            
    except Exception as e:
        error_msg = f"❌ Prediction execution failed. Details: {str(e)}"
        print(f"\n--- Error ---")
        print(error_msg)
        
        if args.verbose:
            traceback.print_exc()
        
        # Prepare error result structure
        error_result = {
            "status": "error",
            "error_message": str(e),
            "input_file": getattr(args, 'feature_dict_path', getattr(args, 'batch_input_csv', 'unknown')),
            "timestamp": time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time() + 8 * 3600))
        }
        
        try:
            output_dir = os.path.dirname(args.output)
            if output_dir:
                os.makedirs(output_dir, exist_ok=True)
            
            with open(args.output, 'w', encoding='utf-8') as f:
                json.dump(error_result, f, indent=4, ensure_ascii=False)
            
            print(f"💾 Error information saved to: {args.output}")
        except Exception as save_error:
            print(f"❌ Failed to save error results: {save_error}")
        
        return 1

if __name__ == "__main__":
    exit_code = main()
    exit(exit_code)