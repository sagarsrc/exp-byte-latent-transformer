# %%
import requests
from collections import Counter
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict, Tuple
from concurrent.futures import ThreadPoolExecutor
import pickle
import os
from prettytable import PrettyTable
import pandas as pd

# Configuration for model storage
MODEL_DIR = os.environ.get("MODEL_DIR", "./models")

# Create the model directory if it doesn't exist
os.makedirs(MODEL_DIR, exist_ok=True)


# %%
def get_multiple_texts() -> str:
    """Fetch multiple texts to create a larger corpus"""
    urls = [
        "https://www.gutenberg.org/files/1661/1661-0.txt",  # Sherlock Holmes
        "https://www.gutenberg.org/files/1342/1342-0.txt",  # Pride and Prejudice
        "https://www.gutenberg.org/files/11/11-0.txt",  # Alice in Wonderland
        "https://www.gutenberg.org/files/84/84-0.txt",  # Frankenstein
    ]

    combined_text = ""
    for url in urls:
        try:
            response = requests.get(url)
            response.raise_for_status()
            combined_text += response.text
        except:
            print(f"Failed to fetch {url}")

    return combined_text


def get_ngram_frequencies(
    text: str, ngram: int, parallel: bool = True
) -> Dict[str, Counter]:
    """Build an n-gram model with improved context handling"""
    cache_file = os.path.join(
        MODEL_DIR, f"ngram_model_{ngram}.pkl"
    )  # Use the model directory

    # Try to load from cache first
    if os.path.exists(cache_file):
        print("Loading n-gram model from cache...")
        with open(cache_file, "rb") as f:
            return pickle.load(f)

    print("Building new n-gram model...")
    ngram_follows = {}
    text_bytes = text.encode("utf-8")
    padded_text = b" " * ngram + text_bytes

    def process_ngram(i: int):
        context = bytes(padded_text[i : i + ngram])
        next_byte = padded_text[i + ngram]
        return context, next_byte

    if parallel:
        with ThreadPoolExecutor() as executor:
            results = list(executor.map(process_ngram, range(len(padded_text) - ngram)))
    else:
        results = [process_ngram(i) for i in range(len(padded_text) - ngram)]

    for context, next_byte in results:
        if context not in ngram_follows:
            ngram_follows[context] = Counter()
        ngram_follows[context][next_byte] += 1

    # Cache the model
    with open(cache_file, "wb") as f:
        pickle.dump(ngram_follows, f)

    return ngram_follows


def calculate_byte_entropy(
    probabilities: List[float], surprise_factor: float = 1.0
) -> float:
    """Calculate entropy with adjustable surprise factor"""
    probs = np.array([p for p in probabilities if p > 0])
    base_entropy = -np.sum(probs * np.log2(probs))
    return base_entropy * surprise_factor


def analyze_sentence_entropy(
    sentence: str,
    ngram_models: Dict[int, Dict[str, Counter]],
) -> List[float]:
    """Calculate entropy by combining analysis from multiple n-gram models"""
    bytes_data = sentence.encode("utf-8")
    max_ngram = max(ngram_models.keys())  # what is the max ngram's model max([3,4,5])?
    padded_data = b" " * max_ngram + bytes_data
    entropies = []

    for i in range(len(bytes_data)):
        entropy_values = []

        # Get entropy from each n-gram model
        for n, model in ngram_models.items():
            context = bytes(padded_data[i : i + n])

            if context in model:
                counter = model[context]
                total = sum(counter.values())
                probs = [count / total for count in counter.values()]
                entropy = -np.sum(np.array(probs) * np.log2(probs))
                entropy_values.append(entropy)

        if entropy_values:
            # Combine entropies resulted from each ngram model (using average)
            combined_entropy = np.mean(entropy_values)
        else:
            # Default entropy for unknown contexts
            combined_entropy = 4.0

        entropies.append(combined_entropy)

    return entropies


def visualize_sentence_entropy(
    sentence: str,
    ngram_models: Dict[int, Dict[str, Counter]],
    threshold: float = 3.0,
    patch_boundaries: List[int] = None,
    method_name: str = "",
):
    """Visualize entropy with combined n-gram models and patch boundaries"""
    entropies = analyze_sentence_entropy(sentence, ngram_models)

    bytes_data = sentence.encode("utf-8")
    chars = [chr(b) if 32 <= b <= 126 else f"<{b}>" for b in bytes_data]

    plt.figure(figsize=(15, 6))
    positions = np.arange(len(entropies))

    plt.plot(positions, entropies, "b-", linewidth=2, label="Combined Entropy")
    plt.axhline(
        y=np.mean(entropies), color="green", linestyle="dotted", label="Average Entropy"
    )

    # Add vertical lines at patch boundaries
    if patch_boundaries:
        for boundary in patch_boundaries:
            plt.axvline(x=boundary, color="purple", linestyle="--", alpha=0.5)

    plt.title(f"Combined N-gram (3,4,5) Entropy Analysis - {method_name}")
    plt.ylabel("Entropy (bits)")
    plt.grid(True, alpha=0.3)

    plt.xticks(
        positions,
        chars,
        rotation=0,
        ha="center",
        fontsize=18,
        fontweight="bold",
        fontname="Helvetica",
    )

    plt.axhline(y=threshold, color="red", linestyle="--", label="Threshold")
    for i, entropy in enumerate(entropies):
        if entropy > threshold:
            plt.plot(i, entropy, "ro", markersize=8)
        else:
            plt.plot(i, entropy, "bo", markersize=4)

    plt.legend()
    plt.tight_layout()
    plt.show()


def detect_patches_global(
    sentence: str, entropies: List[float], threshold: float = None
) -> List[str]:
    """
    Detect patches using global threshold constraint.
    """
    if threshold is None:
        threshold = np.mean(entropies) + 1.5 * np.std(entropies)

    bytes_data = sentence.encode("utf-8")
    chars = [chr(b) if 32 <= b <= 126 else f"<{b}>" for b in bytes_data]

    patches = []
    current_patch = []
    in_high_entropy = False

    for i, (char, entropy) in enumerate(zip(chars, entropies)):
        if entropy > threshold:
            if not in_high_entropy:
                if current_patch:
                    patches.append("".join(current_patch))
                current_patch = [char]
                in_high_entropy = True
            else:
                current_patch.append(char)
        else:
            if in_high_entropy:
                if current_patch:
                    patches.append("".join(current_patch))
                current_patch = [char]
                in_high_entropy = False
            else:
                current_patch.append(char)

    if current_patch:
        patches.append("".join(current_patch))

    return patches


def detect_patches_monotonic(
    sentence: str, entropies: List[float], theta_r: float = 0.5
) -> List[str]:
    """
    Detect patches using approximate monotonic constraint
    """
    bytes_data = sentence.encode("utf-8")
    chars = [chr(b) if 32 <= b <= 126 else f"<{b}>" for b in bytes_data]

    bytes_data = sentence.encode("utf-8")
    chars = [chr(b) if 32 <= b <= 126 else f"<{b}>" for b in bytes_data]

    patches = []
    current_patch = []

    for i in range(len(chars)):
        current_patch.append(chars[i])

        # Check if entropy drops significantly at this point
        if i < len(entropies) - 1:
            entropy_drop = entropies[i] - entropies[i + 1]
            if entropy_drop > theta_r:
                patches.append("".join(current_patch))
                current_patch = []

    # Add remaining characters as final patch
    if current_patch:
        patches.append("".join(current_patch))

    return patches


def detect_patches_double_diff(sentence: str, entropies: List[float]) -> List[str]:
    """
    Detect patches using second derivative (difference of differences) of entropy.
    Creates patches only when entropy rises above the sentence's average entropy.

    Args:
        sentence: Input text string
        entropies: List of entropy values for each character

    Returns:
        List of detected patches
    """
    bytes_data = sentence.encode("utf-8")
    chars = [chr(b) if 32 <= b <= 126 else f"<{b}>" for b in bytes_data]

    # Calculate average entropy for the sentence
    avg_entropy = np.mean(entropies)

    patches = []
    current_patch = []
    was_below = True  # Start assuming we were below average

    for i in range(len(chars)):
        current_patch.append(chars[i])

        if i < len(entropies):
            is_above = entropies[i] > avg_entropy

            # Create new patch only when we transition from below to above average
            if is_above and was_below:
                patches.append("".join(current_patch))
                current_patch = []

            was_below = not is_above

    # Add remaining characters as final patch
    if current_patch:
        patches.append("".join(current_patch))

    return patches


def display_patches_table(sentence: str, patches: List[str], method_name: str):
    """Display patches in a tabular format using PrettyTable"""
    table = PrettyTable()

    # Create column headers
    table.field_names = ["Patch"] + [f"{i+1}" for i in range(len(patches))]

    # Add rows
    table.add_row(["Input"] + patches)
    table.add_row(["#Bytes"] + [len(patch.encode("utf-8")) for patch in patches])

    print(f"\n{method_name}:")
    print(table)


def analyze_and_print_patches(
    sentence: str,
    ngram_models: Dict[int, Dict[str, Counter]],
    threshold: float = None,
):
    """Analyze a sentence using combined n-gram models"""
    print(f"\nAnalyzing: {sentence}")

    # Get entropies from combined models
    entropies = analyze_sentence_entropy(sentence, ngram_models)
    if threshold is None:
        threshold = np.mean(entropies) + 1 * np.std(entropies)

    # Calculate patch boundaries for each method
    def get_patch_boundaries(patches: List[str]) -> List[int]:
        boundaries = []
        current_pos = 0
        for patch in patches[:-1]:  # Skip last patch since it doesn't create a boundary
            current_pos += len(patch)
            boundaries.append(current_pos)
        return boundaries

    # Detect patches using both methods
    global_patches = detect_patches_global(sentence, entropies, threshold)
    monotonic_patches = detect_patches_monotonic(sentence, entropies)
    double_diff_patches = detect_patches_double_diff(sentence, entropies)

    # Display results in tables
    display_patches_table(
        sentence, global_patches, "Combined N-gram Global Threshold Patches"
    )

    # Visualize with monotonic patches
    monotonic_boundaries = get_patch_boundaries(monotonic_patches)
    visualize_sentence_entropy(
        sentence,
        ngram_models,
        threshold,
        monotonic_boundaries,
        "Monotonic Constraint Patches",
    )
    display_patches_table(
        sentence, monotonic_patches, "Combined N-gram Monotonic Constraint Patches"
    )

    # Visualize with double derivative patches
    double_diff_boundaries = get_patch_boundaries(double_diff_patches)
    visualize_sentence_entropy(
        sentence,
        ngram_models,
        threshold,
        double_diff_boundaries,
        "Double Derivative Patches",
    )
    display_patches_table(
        sentence, double_diff_patches, "Combined N-gram Double Derivative Patches"
    )


def show_most_common_ngrams(ngram_model: Dict[bytes, Counter], top_n: int = 10):
    """Display the most common n-gram contexts and their following characters"""
    table = PrettyTable()
    table.field_names = ["Context", "Following Bytes", "Frequency"]

    # Calculate total frequencies for each context
    context_frequencies = {}
    for context, counter in ngram_model.items():
        total_freq = sum(counter.values())
        context_frequencies[context] = total_freq

    # Get top N contexts
    top_contexts = sorted(
        context_frequencies.items(), key=lambda x: x[1], reverse=True
    )[:top_n]

    # Create a DataFrame to store the results
    results = []

    for context, total_freq in top_contexts:
        # Try to decode context if it's ASCII
        try:
            context_str = context.decode("utf-8")
        except UnicodeDecodeError:
            context_str = str(list(context))

        # Get top 3 following bytes for this context
        following = ngram_model[context].most_common(3)
        following_str = ", ".join([f"'{chr(b)}' ({c})" for b, c in following])

        results.append([f"'{context_str}'", following_str, total_freq])

    # Create a DataFrame and print it
    df = pd.DataFrame(results, columns=["Context", "Following Bytes", "Frequency"])
    print("\nMost Common N-gram Patterns:")
    print(df.to_string(index=False))  # Use DataFrame's to_string method


# %%
# Initialize corpus and multiple n-gram models
print("Initializing corpus and n-gram models...")
corpus = get_multiple_texts()

# Create models for n=3,4,5
ngram_models = {}
for n in [3, 4, 5]:
    print(f"Building {n}-gram model...")
    ngram_models[n] = get_ngram_frequencies(corpus, n, parallel=True)
print("Initialization complete!")

# Test different sentences
test_sentences = [
    "I walked to the store and bought a book.",
    "I WALKED to the store and bought @ BOOK!",
    "Sherlock Holmes is a smart detective.",
]

# Analyze each sentence with each n-gram model
for sentence in test_sentences:
    print(f"\n{'='*50}")
    print(f"Analyzing with combined 3,4,5-gram models: {sentence}")
    print(f"{'='*50}")

    analyze_and_print_patches(sentence, ngram_models)
    print("\n" + "=" * 50)

# Show most common patterns for each n-gram model
for n in [3, 4, 5]:
    print(f"\nMost common {n}-gram patterns:")
    show_most_common_ngrams(ngram_models[n])

# %%
