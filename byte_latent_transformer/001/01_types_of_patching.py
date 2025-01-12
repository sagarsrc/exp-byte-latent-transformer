"""
This notebook demonstrates two types of patching: fixed-length and space-based.
"""

# %%
from prettytable import PrettyTable


def fixed_length_patching(input_string, chunk_size):
    # Split string into chunks of fixed size in bytes
    byte_string = input_string.encode("utf-8")  # Convert to bytes
    return [
        (byte_string[i : i + chunk_size]).decode("utf-8")
        for i in range(0, len(byte_string), chunk_size)
    ]


def space_length_patching(input_string):
    # Split string at spaces
    return input_string.split()


def showcase_patching_examples():
    # Create table for fixed length patching
    fixed_table = PrettyTable()
    sample_text = "Hello world! This is a test."
    byte_chunk_size = 5
    fixed_patches = fixed_length_patching(sample_text, byte_chunk_size)

    # Create vertical header
    fixed_table.field_names = ["Patch"] + [f"{i+1}" for i in range(len(fixed_patches))]
    fixed_table.add_row(["Input"] + fixed_patches)
    byte_lengths = [len(patch.encode("utf-8")) for patch in fixed_patches]
    if byte_lengths[-1] < byte_chunk_size:
        byte_lengths[-1] = (
            str(byte_lengths[-1])
            + " + "
            + "pad*"
            + str(byte_chunk_size - byte_lengths[-1])
        )
    fixed_table.add_row(["#Bytes"] + byte_lengths)

    # Create table for space patching
    space_table = PrettyTable()
    sample_text = "Hello world! This is a test."
    space_patches = space_length_patching(sample_text)

    # Create vertical header
    space_table.field_names = ["Patch"] + [f"{i+1}" for i in range(len(space_patches))]
    space_table.add_row(["Input"] + space_patches)
    space_table.add_row(
        ["#Bytes"] + [len(patch.encode("utf-8")) for patch in space_patches]
    )  # Add row for byte count

    print("\nFixed Length Patching (5 bytes per patch):")
    print(fixed_table)
    print("\nSpace-based Patching:")
    print(space_table)


# %%

showcase_patching_examples()
