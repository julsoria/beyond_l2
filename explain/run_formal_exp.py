import argparse
import os
from pathlib import Path
import sys
import torch
import logging
import csv

from cabrnet.core.utils.data import DatasetManager as DM
from subset_minimal_axp import FormalExplanationBase, load_model
from subset_minimal_axp import TopKFormalExplanation, SpatialFormalExplanation
from extended_explainers import (
    CosineSimFormalExplanation, PIPSimplexFormalExplanation, PIPSparseFormalExplanation,
    IsotropicGaussianFormalExplanation, IsotropicLogDistanceFormalExplanation,
)

from tqdm import tqdm, trange
import time
CWD = Path(os.getcwd())

import psutil

ACCEPTED_PARADIGMS = [
    "triangle", "hypersphere", "top_k", "cosine", "cosine_hypersphere",
    "simplex", "pip_sparse", "isotropic_gaussian", "isotropic_gaussian_hypersphere",
    "isotropic_log", "isotropic_log_hypersphere",
]
SPATIAL_PARADIGMS = ["hypersphere", "triangle"]
ACCEPTED_DATASETS = ["stanford_cars", "cub200", "flowers102", "oxford_iiit_pet", "cifar10", "cifar100", "mnist", "stanford_dogs", "tiny_imagenet"]
ACCEPTED_ARCHITECTURES = ["protopnet", "tesnet", "pipnet", "protopool", "protopnet_gaussian_iso"]

logger = logging.getLogger(__name__)


def check_memory_usage(threshold_mb=500):
    """Check current memory usage and raise an error if it exceeds the threshold."""
    process = psutil.Process()
    memory_info = process.memory_info()
    memory_used_mb = memory_info.rss / (1024 * 1024)  # Convert bytes to MB

    if memory_used_mb > threshold_mb:
        print(f"Memory usage exceeded! Current usage: {memory_used_mb:.2f} MB")
        sys.exit("Terminating script due to high memory usage.")
    # else:
    #     print(f"Memory usage is within limits: {memory_used_mb:.2f} MB")


def get_args():
    argparser = argparse.ArgumentParser(description="Run formal explanation.")
    argparser.add_argument(
        "--paradigm",
        type=str,
        default="top_k",
        help=(
            "Paradigm to use for explanation. 'top_k' works for any architecture; "
            "'triangle'/'hypersphere' are the original Euclidean ALE paradigms (ProtoPNet, ProtoPool); "
            "'cosine'/'cosine_hypersphere' target TesNet; 'simplex'/'pip_sparse' target PIP-Net; "
            "'isotropic_gaussian[_hypersphere]'/'isotropic_log[_hypersphere]' target the Isotropic "
            "Gaussian ProtoPNet."
        ),
        choices=ACCEPTED_PARADIGMS,
    )
    argparser.add_argument(
        "--confidence",
        action="store_true",
        help="Use confidence score for explanation.",
    )
    argparser.add_argument(
        "--time_bench",
        action="store_true",
        help="Use time benchmark for explanation.",
    )
    argparser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing explanations (if they exist).",
    )
    argparser.add_argument(
        "--all_indices",
        action="store_true",
        help="Run the explanation for all indices in the test set.",
    )
    argparser.add_argument(
        "--correct-only",
        action="store_true",
        help="Only run explanations for correctly classified samples.",
    )
    argparser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility. Default=42.",
    )
    argparser.add_argument(
        "--data",
        type=str,
        default="stanford_cars",
        help="Dataset to use for explanation. Default=stanford_cars.",
        choices=ACCEPTED_DATASETS
    )
    argparser.add_argument(
        "--arch",
        type=str,
        default="protopnet",
        help=f"Model architecture to use for explanation. Default=protopnet. Options: {ACCEPTED_ARCHITECTURES}",
        choices=ACCEPTED_ARCHITECTURES,
    )
    argparser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to the configuration file. Default=None.",
    )

    args = argparser.parse_args()
    return args




def get_random_idx(dataloader, num_per_class=5, flatten=True):
    """
    Get random indices for each class in the dataloader.
    """
    # Get the number of classes from the dataloader
    if 'classes' in dataloader.dataset.__dict__:
        num_classes = len(dataloader.dataset.classes)
    elif '_labels' in dataloader.dataset.__dict__:
        num_classes = max(dataloader.dataset._labels) + 1
    elif 'targets' in dataloader.dataset.__dict__:
        num_classes = max(dataloader.dataset.targets) + 1
    else:
        raise ValueError("Dataset does not have 'classes', '_labels', or 'targets' attribute to determine number of classes.")
    # Create a list to store the indices for each class
    idx_per_class = [[] for _ in range(num_classes)]
    # Iterate through the dataloader
    for i, (_, label) in enumerate(dataloader):
        # Get the label for the current batch
        label = label.item()
        # Append the index to the corresponding class list
        idx_per_class[label].append(i)
        # If the class list has reached the desired number of indices, break
        if len(idx_per_class[label]) >= num_per_class:
            continue

    #  shuffle the indices for each class
    for class_idx in trange(num_classes):
        torch.manual_seed(0)
        class_indices = torch.tensor(idx_per_class[class_idx])
        # print(f"Class {class_idx} indices: {class_indices}")
        indices = torch.randperm(len(class_indices))
        indices = indices[:num_per_class]
        # print(f"Random indices for class {class_idx}: {class_indices[indices]}")
        idx_per_class[class_idx] = class_indices[indices].tolist()
    # Remove the empty lists from the list of indices
    if flatten:
        # Flatten the list of indices
        idx_per_class = [idx for class_indices in idx_per_class for idx in class_indices]
    else:
        pass
    # Return the indices for each class
    return idx_per_class


def get_latest_iteration(folder_path):
    """
    Get the latest image explained from the folder.
    Format: folder_path/<iteration_number>.txt
    """
    # Get a list of all files in the directory
    files = os.listdir(folder_path)
    # Filter the list to only include files that match the pattern <iteration_number>.txt
    iteration_files = [f for f in files if f.endswith(".txt") and f[:-4].isdigit()]
    # Extract the iteration numbers from the filenames and convert them to integers
    iteration_numbers = [int(f[:-4]) for f in iteration_files]
    # Return the maximum iteration number
    return max(iteration_numbers) if iteration_numbers else None


def get_all_iterations(folder_path):
    """
    Get all the iterations from the folder.
    Format: folder_path/<iteration_number>.txt
    """
    # Get a list of all files in the directory
    files = os.listdir(folder_path)
    # Filter the list to only include files that match the pattern <iteration_number>.txt
    iteration_files = [f for f in files if f.endswith(".txt") and f[:-4].isdigit()]
    # Extract the iteration numbers from the filenames and convert them to integers
    iteration_numbers = [int(f[:-4]) for f in iteration_files]
    # Return the maximum iteration number
    return iteration_numbers if iteration_numbers else None


def exp_check(formal_explanation, model, test_loader, explanation_path, device):
    img, label = next(iter(test_loader))
    iter_idx = 0
    img = img.to(device)
    label = label.to(device)
    predicted_label = torch.argmax(model(img)[0], dim=1)
    formal_explanation.explain(
        img,
        predicted_label,
        # label,
        top_k=True,
        max_only=False,
        triangle_inequality=False,  # TRIANGLE_INEQUALITY,
        hypersphere_approximation=False,  # HYPERSPHERE_APPROXIMATION,
        verbose=False,
    )
    explanations = formal_explanation.explanation

    # Print the explanations
    print("Generated Explanations:")
    # for explanation in explanations:
    #     print(explanation)
    print(list([el[0] for el in explanations]))
    print(f"Explanation length: {len(explanations)}")
    predicted_label = torch.argmax(model(img)[0], dim=1)
    print(f"Predicted label: {predicted_label}")

    explanation_folder = explanation_path / "top_k_explanations"
    with open(explanation_folder / f"{iter_idx}.txt", "r") as f:
        for line in f:
            print(line.strip())


def run_indices(explainer, model, test_loader, explanation_path, device, indices, paradigm: str = "triangle", confidence: bool = False, time_bench: bool = False, overwrite: bool = False, correct_only: bool = False) -> int:
    """
    Run the explanation for the given indices and log to CSV with robust header handling.
    """
    model.eval()

    # --- CSV Setup ---
    # Ensure the parent directory exists
    explanation_path.parent.mkdir(parents=True, exist_ok=True)

    csv_file = explanation_path.parent / f"{paradigm}_explanations.csv"
    file_exists = csv_file.exists()

    # Define Headers
    fieldnames = [
        'Idx', 'True Label', 'Pred Label', 'Correct', 'Exp Size',
        'Conf Score', 'Conf Softmax', '2nd Conf Score', '2nd Conf Softmax', 'Diff Conf Score', 'Diff Conf Softmax', 'Next Label',
        'Runtime (s)'
    ]

    # Auto-Overwrite Check: If headers don't match, force overwrite to prevent CSV corruption
    mode = 'a'
    processed_indices = set()
    if file_exists and not overwrite:
        with open(csv_file, 'r', newline='') as f:
            reader = csv.reader(f)
            try:
                existing_header = next(reader)
                if set(existing_header) != set(fieldnames):
                    print(f"Warning: CSV Header mismatch (Old: {len(existing_header)} cols, New: {len(fieldnames)} cols). Overwriting file.")
                    overwrite = True
                else:
                    if 'Idx' in existing_header:
                        idx_col = existing_header.index('Idx')
                        for row in reader:
                            if row and len(row) > idx_col:
                                try:
                                    processed_indices.add(int(row[idx_col]))
                                except ValueError:
                                    pass
                    else:
                        print("Warning: 'Idx' column not found in existing CSV header. Cannot track previously processed indices.")
            except StopIteration:
                overwrite = True # File is empty

    if overwrite:
        mode = 'w'

    if len(processed_indices) > 0:
        print(f"Found {len(processed_indices)} previously processed indices. They will be skipped.")

    # Open CSV
    csv_f = open(csv_file, mode, newline='')
    writer = csv.DictWriter(csv_f, fieldnames=fieldnames)

    # Write header if we are creating a new file or overwriting
    if overwrite or not file_exists:
        writer.writeheader()
    # -----------------

    if time_bench:
        print("Time benchmark is enabled.")
        total_time = 0

    n_samples = 0

    # Iterate
    for idx, (img, label) in tqdm(enumerate(test_loader), total=len(test_loader)):
        # check_memory_usage(threshold_mb=5000) # Optional: Enable if needed

        if idx not in indices and not confidence:
            continue

        if idx in processed_indices:
            continue

        img = img.to(device)
        label = label.to(device)

        with torch.no_grad():
            output = model(img)
            logits = output[0] # Shape (K)

            # Get Top 2 Logits
            top2_logits, top2_indices = torch.topk(logits, k=2)

            # Get Softmax Probabilities
            probs = torch.softmax(logits, dim=-1)
            top2_probs, _ = torch.topk(probs, k=2)

            predicted_label = top2_indices[0][0].item()
            true_label = label.item()
            is_correct_pred = (predicted_label == true_label)

        if correct_only and not is_correct_pred:
            continue

        # Prepare Row Data (Initialize with defaults)
        row_data = {
            'Idx': idx,
            'True Label': true_label,
            'Pred Label': predicted_label,
            'Correct': is_correct_pred,
            'Exp Size': None,

            # Add New Metrics
            'Conf Score': top2_logits[0][0].item(),
            'Conf Softmax': top2_probs[0][0].item(),
            '2nd Conf Score': top2_logits[0][1].item(),
            '2nd Conf Softmax': top2_probs[0][1].item(),
            'Diff Conf Score': (top2_logits[0][0] - top2_logits[0][1]).item(),
            'Diff Conf Softmax': (top2_probs[0][0] - top2_probs[0][1]).item(),
            'Next Label': top2_indices[0][1].item(),
            'Runtime (s)': None,
        }

        try:
            # ALWAYS measure time for the CSV
            sample_start_time = time.time()

            # Standard execution
            explainer.explain(img, label, verbose=False)

            if isinstance(explainer.explanation, dict):
                row_data['Exp Size'] = len(explainer.explanation)
            else:
                row_data['Exp Size'] = len(explainer.explanation) if explainer.explanation else 0

            # Calculate elapsed time
            elapsed_time = time.time() - sample_start_time
            row_data['Runtime (s)'] = round(elapsed_time, 4)

            if time_bench:
                total_time += elapsed_time

            n_samples += 1

            # Write immediately
            writer.writerow(row_data)
            csv_f.flush()

        except Exception as e:
            print(f"Error processing index {idx}: {e}")
            exit(1)
            writer.writerow(row_data)
            continue

    csv_f.close()
    print(f"Completed explanations for {n_samples} samples. Results saved to {csv_file}")

    if time_bench:
        if len(indices) > 0:
            avg_time = total_time / len(indices)
        else:
            avg_time = 0
        print(f"Total time taken: {total_time:.2f} seconds")
        print(f"Average time per sample: {avg_time:.4f} seconds")

    return n_samples


def check_indices(indices, path):
    """
    For all indices, check if the file exists in the path.
    Return True if all files exist, False otherwise.
    """
    # create the path and parent directories if they do not exist
    if not os.path.exists(path):
        os.makedirs(path)
    files = os.listdir(path)
    for idx in indices:
        file_name = f"{idx}.txt"
        if file_name not in files:
            print(f"File {file_name} does not exist.")
            return False
    return True


def measure_accuracy(model, testloader, indices):
    total = 0
    correct = 0
    model.eval()
    device = next(model.parameters()).device

    # Assuming testloader is a DataLoader object
    for i, (img, label) in enumerate(testloader):
        if i not in indices:
            continue
        # Forward pass
        img = img.to(device)
        label = label.to(device)
        output = model(img)[0]
        predicted = torch.argmax(output, 1)
        # Calculate accuracy
        # Assuming label is a tensor of the same size as predicted
        total += label.size(0)
        correct += (predicted == label).sum().item()
    accuracy = correct / total
    return accuracy

def main(paradigm="triangle", confidence=False, time_bench=False, overwrite=False, all_indices=False, seed=42, data="stanford_cars", arch="protopnet", config=None, correct_only=False) -> int:
    """
    Main function to run the explanation.
    """
    if data not in ACCEPTED_DATASETS:
        raise ValueError(f"Data {data} is not valid.")

    if data == "stanford_cars":
        final_model_path = CWD / 'logs' / f'protopnet_stanford_cars_{seed}' / 'final'
    else:
        # Fallback generic
        final_model_path = CWD / 'logs' / f'{arch}_{data}_{seed}' / 'final'

    if config is not None:
        final_model_path = Path(config) / 'final'

    if not final_model_path.exists():
        # Basic error handling from original
        print(f"Path not found: {final_model_path}")
        return 0

    prototype_path = final_model_path / "prototypes.pth"
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    explanation_path = final_model_path / "explanations"
    if not explanation_path.exists():
        explanation_path.mkdir(parents=True, exist_ok=True)

    model, test_loader = load_model(final_model_path, seed=seed, device=device, test_set=True)

    if test_loader.batch_size != 1:
        test_loader = torch.utils.data.DataLoader(test_loader.dataset, batch_size=1, shuffle=False)

    loader_to_use = test_loader
    subdir = "explanations"

    explanation_path = final_model_path / subdir
    if not explanation_path.exists():
        explanation_path.mkdir(parents=True, exist_ok=True)

    # --- CLASS SELECTION LOGIC ---
    if paradigm == "cosine":
        print("Initializing CosineSimFormalExplanation (Triangle)...")
        explanation_cls = CosineSimFormalExplanation
        formal_explanation = explanation_cls(model, device, prototype_filepath=prototype_path, save_proto=True, load_proto=False, paradigm="triangle")

    elif paradigm == "cosine_hypersphere":
        print("Initializing CosineSimFormalExplanation (Spherical Hypersphere)...")
        explanation_cls = CosineSimFormalExplanation
        formal_explanation = explanation_cls(model, device, prototype_filepath=prototype_path, save_proto=True, load_proto=False, paradigm="hypersphere")
    elif paradigm == "simplex":
        print("Initializing SimplexFormalExplanation...")
        explanation_cls = PIPSimplexFormalExplanation
        formal_explanation = explanation_cls(model, device, prototype_filepath=prototype_path, save_proto=True, load_proto=False)

    elif paradigm == "pip_sparse":
        print("Initializing PIPSparseFormalExplanation...")
        explanation_cls = PIPSparseFormalExplanation
        formal_explanation = explanation_cls(model, device, prototype_filepath=prototype_path, save_proto=True, load_proto=False)

    # --- ISOTROPIC GAUSSIAN EXPLAINERS ---
    elif paradigm == "isotropic_gaussian":
        print("Initializing IsotropicGaussianFormalExplanation (Triangle Inequality)...")
        explanation_cls = IsotropicGaussianFormalExplanation
        formal_explanation = explanation_cls(model, device, prototype_filepath=prototype_path, save_proto=True, load_proto=False, paradigm="triangle")

    elif paradigm == "isotropic_gaussian_hypersphere":
        print("Initializing IsotropicGaussianFormalExplanation (Euclidean Hypersphere)...")
        explanation_cls = IsotropicGaussianFormalExplanation
        formal_explanation = explanation_cls(model, device, prototype_filepath=prototype_path, save_proto=True, load_proto=False, paradigm="hypersphere")

    elif paradigm == "isotropic_log":
        print("Initializing IsotropicLogDistanceFormalExplanation (Triangle Inequality)...")
        explanation_cls = IsotropicLogDistanceFormalExplanation
        formal_explanation = explanation_cls(model, device, prototype_filepath=prototype_path, save_proto=True, load_proto=False, paradigm="triangle")

    elif paradigm == "isotropic_log_hypersphere":
        print("Initializing IsotropicLogDistanceFormalExplanation (Euclidean Hypersphere)...")
        explanation_cls = IsotropicLogDistanceFormalExplanation
        formal_explanation = explanation_cls(model, device, prototype_filepath=prototype_path, save_proto=True, load_proto=False, paradigm="hypersphere")

    elif paradigm in SPATIAL_PARADIGMS:
        formal_explanation = SpatialFormalExplanation(model, device, prototype_filepath=prototype_path, save_proto=True, load_proto=False, paradigm=paradigm)

    elif paradigm == "top_k":
        formal_explanation = TopKFormalExplanation(model, device, prototype_filepath=prototype_path, save_proto=True, load_proto=False)
    else:
        raise ValueError(f"Paradigm {paradigm} is not valid.")
    # -----------------------------

    # [Keep Index Loading Logic Unchanged]
    if all_indices:
        random_idx = list(range(len(loader_to_use)))
    else:
        random_idx_file = final_model_path / "random_indices.txt"
        if random_idx_file.exists():
            with open(random_idx_file, "r") as f:
                random_idx = [int(line.strip()) for line in f.readlines()]
            print(f"Loaded {len(random_idx)} random indices from file.")
        else:
            print(f"File {random_idx_file} does not exist. Generating random indices.")
            numel = 5
            if data == "cub200":
                numclasses = 200
            elif data == "stanford_cars":
                numclasses = 196
            elif data == "flowers102":
                numclasses = 102
            elif data == "oxford_iiit_pet":
                numclasses = 37
            elif data.startswith("cifar") and data[5:].isdigit():
                numclasses = int(data[5:])
                assert numclasses in [10, 100], f"Invalid number of classes for CIFAR: {numclasses}"
            elif data == "mnist":
                numclasses = 10
            elif data == "stanford_dogs":
                numclasses = 120
            elif data == "tiny_imagenet":
                numclasses = 200
            else:
                raise ValueError(f"Data {data} is not valid. Options are: {ACCEPTED_DATASETS}")
            random_idx = get_random_idx(test_loader, num_per_class=numel)

            assert len(random_idx) == numel*numclasses, f"Expected 5 indices, got {len(random_idx)}"
            print(f"Random indices: {random_idx[:10]}")
            with open(random_idx_file, "w") as f:
                for iidx in random_idx:
                    f.write(f"{iidx}\n")

            print(f"Saved {len(random_idx)} random indices to file {random_idx_file}.")

        print(f"Saved {len(random_idx)} random indices to file {random_idx_file}.")

    # Run
    print(f"Running experiments (Correct Only: {correct_only})...")
    n_samples = run_indices(formal_explanation, model, loader_to_use, explanation_path, device, random_idx, paradigm=paradigm, confidence=confidence, time_bench=time_bench, overwrite=overwrite, correct_only=correct_only)

    return n_samples



if __name__ == "__main__":
    args = get_args()
    print(args)

    print(f"Running experiment the following settings:\n{args}")

    start_time = time.time()
    n_samples = main(
        paradigm=args.paradigm,
        confidence=args.confidence,
        time_bench=args.time_bench,
        overwrite=args.overwrite,
        all_indices=args.all_indices,
        seed=args.seed,
        data=args.data,
        arch=args.arch,
        config=args.config,
        correct_only=args.correct_only,
    )

    end_time = time.time()
    total_time = end_time - start_time
    total_memory = psutil.Process().memory_info().rss / (1024 * 1024)
    print(f"Total memory used: {total_memory:.2f} MB")
    print(f"Total time taken: {total_time:.4f} seconds")
    print(f"Total time taken: {total_time / 60:.4f} minutes ({total_time / 3600:.4f} hours)")
    print(f"Average time per sample: {total_time / n_samples:.4f} seconds" if n_samples > 0 else "No samples processed.")
    if n_samples > 0:
        print(f"Number of samples processed: {n_samples}")
    print("Formal explanation completed.")
