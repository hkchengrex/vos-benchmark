# Simple Vidoe Object Segmentation Benchmarking 

## Quick Start

### Installation

#### Locally (recommended):
```bash
git clone https://github.com/hkchengrex/vos-benchmark
pip install -e vos-benchmark
```

#### Via pip (as a library):
`pip install vos-benchmark`

### Using it as a script:
(from the vos-benchmark root directory)

```bash
python benchmark.py -g <path to ground-truth directory> -m <path to prediction directory> -n <number of processes, 16 by default>
```

### Using it as a library:

```python
from vos_benchmark.benchmark import benchmark

# both arguments are passed as a list -- multiple datasets can be specified
benchmark([path to ground-truth directory], [<path to prediction directory>])
```
See `benchmark.py` for an example, and see `vos_benchmark/benchmark.py` for the function signature with additional options.

A `results.csv` will be saved in the prediction directory. 

## Background

I built this tool to accelerate evaluations (J&F) on different video object segmentation benchmarks. Previously, [davis2017-evaluation](https://github.com/davisvideochallenge/davis2017-evaluation) is used, which has several limitations:

- It is slow. Evaluating predictions on the validation set of DAVIS-2017 (30 videos) takes 73.3 seconds. It would take longer for any larger dataset. Ours takes 5.36 seconds (with 16 threads).
- It is tailored to the DAVIS dataset. Evaluating other datasets (like converted OVIS, UVO, or the long video dataset) requires mocking them as DAVIS (setting up "split" text files and following a non-trivial file structure). We don't care. We just take the paths to two folders (ground-truth and predictions) as input.
- It does not work with non-continuous object IDs. We do.

I have tested this script on DAVIS-16/17 and confirmed that it produces identical results as the official evaluation script. 

## Technical Details / Troubleshooting
1. This benchmarking script is simple and dumb. It does not intelligently resolve input problems. If something does not work, most likely the input is problematic. Garbage in, garbage out. Check your input (see below).
We read the input masks using `Image.open` from PIL. Paletted png files and grayscale png should both work.
2. We determine the objects in a frame (ground-truth or prediction) with `np.unique`. If there are any types of antialiasing, blurring, smoothing, etc., that spawn new pixel values, this will not work.
3. From the start of the video, we keep a list of all objects that are seen in either ground-truth or prediction. This is to support datasets where some ground-truth objects appear later in the frame. Predicting objects that are not in the ground-truth harms the final score.
4. By default, we skip the first and the last frame during evaluation. This is in line with the standard semi-supervised video object segmentation evaluation protocol in DAVIS. This can be overridden by specifying `-d` or `--do_not_skip_first_and_last_frame`, or passing `skip_first_and_last=False` (if used as a package).
5. By default, we don't care if all the videos in the ground-truth folder have corresponding predictions. This is to support datasets that contain videos from different splits (e.g., DAVIS puts train/val splits together) in a single folder. If the prediction only contains videos from the validation set, we would only evaluate those videos. This can be overridden by specifying `-s` or `--strict`, or passing `strict=True` (if used as a package). In the strict mode, an exception would be thrown if the sets of videos do not match.
6. If a video is being evaluated, all the frames in the ground-truth folder must have corresponding predictions. Predictions that do not have corresponding ground-truths are simply ignored.

## Related projects:

Official DAVIS 2017 evaluation implementation: https://github.com/davisvideochallenge/davis2017-evaluation

BURST benchmark (evaluates HOTA which is not supported here): https://github.com/Ali2500/BURST-benchmark

TrackEval (a powerful tool with more functionalities): https://github.com/JonathonLuiten/TrackEval


### My video object segmentation projects:

DEVA (latest): https://github.com/hkchengrex/Tracking-Anything-with-DEVA

XMem: https://github.com/hkchengrex/XMem

STCN: https://github.com/hkchengrex/STCN

MiVOS: https://github.com/hkchengrex/MiVOS
