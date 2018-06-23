import os
import sys
import time
from argparse import ArgumentParser
sys.path.append("./src")
from src/faceAverage import Averager


if __name__ == '__main__' :
    start = time.time()

    parser = ArgumentParser()
    parser.add_argument('-i', '--input-dir', dest="input", help="Specify the input directory", type=str, required=True)
    parser.add_argument('-ow', '--output-width', dest="width", help="Specify the output file width. Default 300", type=int)
    parser.add_argument('-oh', '--output-height', dest="height", help="Specify the output file height. Default 400", type=int)
    parser.add_argument('-e', '--extensions', dest="ext", help="Specify the file extensions like *.jpg *.whatevs.file", nargs="+")
    parser.add_argument('-o', '--output-path', dest="output", help="Specify the output path for writing result. Default ./results/[input-dir-name].jpg", type=str)
    parser.add_argument('-w', '--window', dest="window", help="Shows window if set to True", action="store_true", default=False)
    parser.add_argument('-nw', '--no-warps', dest="noWarps", help="Shows warping stage if set to True", action="store_true", default=False)
    parser.add_argument('-nc', '--no-caching', dest="noCaching", help="Load shape features from cache of .txt files", action="store_true", default=False)
    parser.add_argument('-wt', '--window-time', dest="windowTime", help="Specify how long should window stay on in miliseconds", type=int, default=500)

    options = parser.parse_args()
    print(options)
    ext = options.ext or ["*.jpg", "*.jpeg"]
    Averager().run(path=options.input, ext=ext, window=options.window, showWarps=not options.noWarps, windowTime=options.windowTime, useCaching=not options.noCaching).save(name=options.output)

    print(f">>> Executed in {time.time()-start:.2f} seconds")
