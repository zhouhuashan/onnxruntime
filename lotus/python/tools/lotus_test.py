import argparse
import lotus
import numpy as np
import sys
from timeit import default_timer as timer

# simple test program for loading onnx model, feeding all inputs and running the model num_iters times.
def main():
    parser = argparse.ArgumentParser(description='Simple Lotus Test Tool.')
    parser.add_argument('model_path', help='model path')
    parser.add_argument('num_iters', nargs='?', type=int, default=1000, help='model run iterations. default=1000')
    args = parser.parse_args()
    iters = args.num_iters

    sess = lotus.InferenceSession(args.model_path)

    feeds = {}
    for input_meta in sess.get_inputs():
        feeds[input_meta.name] = np.random.rand(*input_meta.shape).astype('f') # assume inputs are float

    start = timer()
    for i in range(iters):
        sess.run([], feeds) # fetch all outputs
    end = timer()

    print('latency: {} ms'.format(((end - start)*1000)/iters))
    return 0

if __name__ == "__main__":
    sys.exit(main())
