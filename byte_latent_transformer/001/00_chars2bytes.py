# %%
def string_to_bytestream(string):
    return bytes(string, "utf-8")


def print_bytestream_representation(input_string):
    bytestream_representation = string_to_bytestream(input_string)
    for s, b in zip(list(input_string), bytestream_representation):
        print(f"{s.replace(' ', '␣'):>2} -> {b:08b} -> {b:>3}")

    print("Bytestream representation:\n", list(bytestream_representation))


# Example usage
if __name__ == "__main__":
    input_string = "Hello Bytes World!"  # You can change this to any word or sentence
    print_bytestream_representation(input_string)
