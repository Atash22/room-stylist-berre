# Run this BEFORE closing your Colab session — captures everything you'll
# need for the slides that isn't already in benchmark_results.json.

import json

full_results = {
    "resnet18": {
        "accuracy": resnet_results["accuracy"],
        "macro_f1": resnet_results["macro_f1"],
        "inference_ms_per_image": resnet_results["inference_ms_per_image"],
        "confusion_matrix": resnet_results["confusion_matrix"],
        "classification_report": resnet_results["report"],
    },
    "efficientnet_b0": {
        "accuracy": effnet_results["accuracy"],
        "macro_f1": effnet_results["macro_f1"],
        "inference_ms_per_image": effnet_results["inference_ms_per_image"],
        "confusion_matrix": effnet_results["confusion_matrix"],
        "classification_report": effnet_results["report"],
    },
    "clip_zero_shot": clip_results,
    "class_names": class_names,
    "dataset_sizes": {
        "train": len(train_ds),
        "val": len(val_ds),
        "test": len(test_ds),
    },
}

with open("/content/full_benchmark_results.json", "w") as f:
    json.dump(full_results, f, indent=2)

# Also save a plain-text version of the reports, easier to paste into slides
with open("/content/classification_reports.txt", "w") as f:
    f.write("=== ResNet-18 ===\n")
    f.write(resnet_results["report"])
    f.write("\n\n=== EfficientNet-B0 ===\n")
    f.write(effnet_results["report"])
    f.write("\n\n=== CLIP zero-shot ===\n")
    f.write(f"accuracy={clip_results['accuracy']:.4f}  macro_f1={clip_results['macro_f1']:.4f}  "
             f"latency={clip_results['inference_ms_per_image']:.1f}ms/img\n")
    f.write("(CLIP zero-shot has no per-class report — no training/fine-tuning was done)\n")

from google.colab import files
files.download("/content/full_benchmark_results.json")
files.download("/content/classification_reports.txt")

print("Downloaded full_benchmark_results.json and classification_reports.txt")
