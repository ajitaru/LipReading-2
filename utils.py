import numpy as np
from pathlib import Path
import random
import h5py

def batch_gen(batch_size):
    data_path = Path.cwd().joinpath('data', 'hdf5')
    files = [x for x in data_path.glob('*.hdf5')]
    #files = files[:10] # for testing only
    with h5py.File(files[0], 'r') as f:
        # Extract the shape of the 'frames' and 'spectrogram' tensors
        frames_shape = f['frames'].shape
        frames_shape = (batch_size, *frames_shape)
        spec_shape = f['spectrogram'].shape
        spec_shape = (batch_size, *spec_shape)

    # Initialize numpy arrays
    vid_in = np.zeros(shape=frames_shape, dtype=np.float32)
    aud_in = np.zeros(shape=spec_shape, dtype=np.float32)

    num_batches = int(np.floor(len(files)/batch_size))
    while True:
        for n in range(num_batches):
            batch_files = files[n*batch_size:(n+1)*batch_size]
            for idx, fpath in enumerate(batch_files):
                with h5py.File(fpath, 'r') as f:
                    vid_in[idx] = f['frames']
                    aud_in[idx] = f['spectrogram']
            yield vid_in, aud_in

        random.shuffle(files)

if __name__ == '__main__':

    gen = batch_gen(2)
    x, y = next(gen)
    print(x.shape, y.shape)

    x, y = next(gen)
    print(x.shape, y.shape)

def print_tensor_shape(tensor, string):
    '''Function to print the tensor shape. Debugging
    :param tensor: Tensor object to describe
    :param string: Debugging string
    :return None
    '''
    if __debug__:
        print('DEBUG: ' + string, tensor.get_shape())