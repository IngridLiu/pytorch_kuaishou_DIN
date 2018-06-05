import os
import pickle


# save serise data
def save_serise_data(data, save_path):
    n_bytes = 2 ** 31
    max_bytes = 2 ** 31 - 1
    data = bytearray(n_bytes)

    ## write
    bytes_out = pickle.dumps(data)
    with open(save_path, 'wb') as file:
        for idx in range(0, n_bytes, max_bytes):
            file.write(bytes_out[idx:idx + max_bytes])

def load_series_data(load_path):
    n_bytes = 2 ** 31
    max_bytes = 2 ** 31 - 1
    data = bytearray(n_bytes)

    ## read
    bytes_in = bytearray(0)
    input_size = os.path.getsize(load_path)
    with open(load_path, 'rb') as file:
        for _ in range(0, input_size, max_bytes):
            bytes_in += file.read(max_bytes)
    data = pickle.loads(bytes_in)
    return data