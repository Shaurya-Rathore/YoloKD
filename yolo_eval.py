from ultralytics import YOLO
import torch
import logging

def evaluate_yolo_model(weights_path, data_yaml, task='detect'):
    # Load model
    model = YOLO(weights_path)
    
    # Run validation on test set
    metrics = model.val(data=data_yaml, 
                      split='test',  # Use test split
                      verbose=True,   # Print progress
                      save_json=True, # Save results to JSON
                      save_conf=True, # Save confidences
                      plots=False)    # Don't generate plots
    
    # Extract key metrics
    results = {
        'metrics/precision(B)': metrics.box.map,    # Mean Average Precision
        'metrics/recall(B)': metrics.box.mar,       # Mean Average Recall
        'metrics/mAP50(B)': metrics.box.map50,     # mAP at IoU=0.5
        'metrics/mAP50-95(B)': metrics.box.map75,  # mAP at IoU=0.5:0.95
        'fitness': metrics.fitness,                 # Overall fitness score
    }
    results['per_class'] = {
        'precision': metrics.box.maps,  # Per-class precision
        'recall': metrics.box.mar_per_class  # Per-class recall
    }
               
    return results

weights_path = "/kaggle/input/spdyolon-weights/best.pt"
data_yaml = "/kaggle/input/waiddataset/WAID-main/WAID-main/WAID/data.yaml"

# Run evaluation
results = evaluate_yolo_model(weights_path, data_yaml)

# Print key metrics
print("\nEvaluation Results:")
print(f"mAP@0.5: {results['metrics/mAP50(B)']:.4f}")
print(f"mAP@0.5:0.95: {results['metrics/mAP50-95(B)']:.4f}")
print(f"Precision: {results['metrics/precision(B)']:.4f}")
print(f"Recall: {results['metrics/recall(B)']:.4f}")
print(f"Overall Fitness: {results['fitness']:.4f}")
