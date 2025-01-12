"""
This code serves as an example for calculating entropy in a text corpus.
It demonstrates the following steps:

1. Fetch Texts: Retrieves multiple texts from online sources to create a larger corpus.
2. Build n-gram Model: Constructs an n-gram model to analyze the frequency of character sequences in the corpus.
3. Calculate Entropy: Computes the entropy for each character position in a given sentence based on the n-gram model.
4. Detect Patches: Identifies segments of the sentence where the entropy is monotonic (either increasing or decreasing).
5. Visualize Results: Plots the entropy values, with options to display threshold and average lines for better analysis.
6. Analyze Sentences: Processes test sentences to analyze their entropy and print detected patches along with visualizations.
"""

# %%
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
from typing import List, Dict
import requests
import os
import pickle
from concurrent.futures import ThreadPoolExecutor

# Configuration for model storage
# Directory where the n-gram model will be stored
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


def analyze_sentence_entropy(
    sentence: str, ngram_model: Dict[bytes, Counter], ngram: int = 2
) -> List[float]:
    """Calculate entropy for each character position"""
    bytes_data = sentence.encode("utf-8")
    padded_data = b" " * ngram + bytes_data
    entropies = []

    for i in range(len(bytes_data)):
        context = bytes(padded_data[i : i + ngram])

        if context in ngram_model:
            counter = ngram_model[context]
            total = sum(counter.values())
            probs = [count / total for count in counter.values()]
            entropy = -np.sum(np.array(probs) * np.log2(probs))
        else:
            entropy = 4.0  # Default entropy for unknown contexts

        entropies.append(entropy)

    return entropies


def detect_patches_monotonic(sentence: str, entropies: List[float]) -> List[str]:
    """Detect patches using monotonic constraint"""
    patches = []
    current_patch = ""
    bytes_data = sentence.encode("utf-8")

    for i in range(len(entropies)):
        current_patch += chr(bytes_data[i])

        # End patch if we're at the last character or entropy increases
        if i == len(entropies) - 1 or entropies[i] < entropies[i + 1]:
            patches.append(current_patch)
            current_patch = ""

    return patches


def detect_patches_global(
    sentence: str, entropies: List[float], threshold: float = None
) -> List[str]:
    """Detect patches using global threshold constraint based on byte-wise entropy"""
    if threshold is None:
        threshold = np.mean(entropies) + 1.0 * np.std(entropies)

    print(f"Threshold: {threshold}")

    bytes_data = sentence.encode("utf-8")

    # Print entropy for each byte
    print("\nByte-wise entropy:")
    for b, entropy in zip(bytes_data, entropies):
        char_repr = chr(b) if 32 <= b <= 126 else f"<{b}>"
        print(
            f"Byte: {b} ('{char_repr}'), Entropy: {entropy:.2f}, Above Threshold: {entropy > threshold}"
        )

    # Find indices where entropy exceeds threshold
    high_entropy_indices = [i for i, e in enumerate(entropies) if e > threshold]
    print(f"\nHigh entropy byte indices: {high_entropy_indices}")

    if not high_entropy_indices:
        return [sentence]

    # Create patches using high entropy bytes as boundaries
    patches = []
    start = 0

    for idx in high_entropy_indices:
        # Add segment before the high entropy byte
        if idx > start:
            patches.append(sentence[start:idx])
        # Add the high entropy byte as its own patch
        patches.append(sentence[idx : idx + 1])
        start = idx + 1

    # Add the remaining segment if any
    if start < len(sentence):
        patches.append(sentence[start:])

    return [p for p in patches if p]


def display_patches_table(sentence: str, patches: List[str], method_name: str):
    """Display patches in a tabular format"""
    print(f"\n{method_name}:")
    for i, patch in enumerate(patches, 1):
        print(f"Patch {i}: '{patch}' ({len(patch.encode('utf-8'))} bytes)")


def visualize_sentence_entropy(
    sentence: str,
    ngram_model: Dict[bytes, Counter],
    ngram: int = 2,
    show_threshold: bool = True,
    show_average: bool = True,
):
    """Visualize byte-wise entropy analysis"""
    entropies = analyze_sentence_entropy(sentence, ngram_model, ngram)
    bytes_data = sentence.encode("utf-8")

    plt.figure(figsize=(15, 6))
    positions = np.arange(len(entropies))

    plt.plot(positions, entropies, "b-", linewidth=2, label="Byte Entropy")

    if show_average:
        average = np.mean(entropies)
        plt.axhline(
            y=average,
            color="green",
            linestyle="dotted",
            label=f"Average Byte Entropy (μ = {average:.2f})",
        )

    if show_threshold:
        threshold = np.mean(entropies) + 1.0 * np.std(entropies)
        plt.axhline(
            y=threshold,
            color="red",
            linestyle="--",
            label=f"Threshold (μ+σ = {threshold:.2f})",
        )

    plt.title(f"{ngram}-gram Byte-wise Entropy Analysis")
    plt.ylabel("Entropy (bits)")
    plt.grid(True, alpha=0.3)

    # X-axis labels showing both byte value and character representation
    x_labels = [
        f"{b}\n'{chr(b)}'" if 32 <= b <= 126 else f"{b}\n<{b}>" for b in bytes_data
    ]
    plt.xticks(positions, x_labels, rotation=0, ha="center", fontsize=12)

    for i, entropy in enumerate(entropies):
        if entropy > threshold:
            plt.plot(i, entropy, "ro", markersize=8)
        else:
            plt.plot(i, entropy, "bo", markersize=4)

    plt.legend()
    plt.tight_layout()
    plt.show()


def analyze_and_print_patches(
    sentence: str,
    ngram_model: Dict[bytes, Counter],
    ngram: int = 2,
    threshold: float = 3,
    show_threshold: bool = True,
    show_average: bool = True,
):
    """Analyze a sentence and print patches using both methods"""
    print(f"\nAnalyzing: {sentence}")

    # Get entropies
    entropies = analyze_sentence_entropy(sentence, ngram_model, ngram)

    # Detect patches using both methods
    global_patches = detect_patches_global(sentence, entropies, threshold)
    monotonic_patches = detect_patches_monotonic(sentence, entropies)

    # Print results
    display_patches_table(sentence, global_patches, "Global Threshold Patches")
    display_patches_table(sentence, monotonic_patches, "Monotonic Constraint Patches")

    # Visualize
    visualize_sentence_entropy(
        sentence,
        ngram_model,
        ngram,
        show_threshold=show_threshold,
        show_average=show_average,
    )


# %%
# Initialize and test
print("Initializing corpus and 2-gram model...")
corpus = get_multiple_texts()
ngram_model = get_ngram_frequencies(corpus, ngram=2)

# %%

# Test sentences
test_sentences = [
    "I walked to the store and bought a book.",
    "I WALKED to the store and bought @ BOOK!",
]

# %%
# Analyze each sentence
for sentence in test_sentences:
    analyze_and_print_patches(
        sentence,
        ngram_model,
        threshold=None,  # set for global patching else mu+1*sigma (entropy)
        show_threshold=True,  # Set to False to hide threshold line
        show_average=True,  # Set to False to hide average line
    )
