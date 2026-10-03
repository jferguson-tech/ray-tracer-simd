# Path Tracer

[![build](https://github.com/jferguson-tech/ray-tracer-simd/actions/workflows/build.yml/badge.svg)](https://github.com/jferguson-tech/ray-tracer-simd/actions/workflows/build.yml)

High-performance C++ path tracer with SIMD acceleration.

![Path Tracer Demo](pathtracer.gif)

## Features
- AVX2 SIMD acceleration: measured 1.9-2.2x whole-frame speedup (3-5x in the vectorized subsystems)
- 8-wide packet ray marching for volumetric shadow rays, vectorized caustics/noise kernels, SSE-backed Vec3
- Multi-threaded tile renderer (std::thread)
- Voxel world with procedurally textured terrain, refractive and reflective water, and emissive light blocks
- Sun and sky lighting, water caustics and volumetric light shafts
- Camera path recording, playback, benchmarking and offline rendering (JSON)
- Repeatable renders: the same command produces the same image, on any number of threads

## Performance

Measured at 128 samples/pixel, offline mode, AMD Ryzen AI 9 HX 370 (12C/24T, MSVC /O2 /arch:AVX2):

| View | Resolution | Scalar | SIMD | Speedup |
|---|---|---|---|---|
| Lake | 640x360 | 26.3 s | 14.2 s | 1.85x |
| Lake | 1280x720 | 104.8 s | 55.0 s | 1.90x |
| River valley | 640x360 | 40.4 s | 18.1 s | 2.23x |
| River valley | 1280x720 | 160.6 s | 71.6 s | 2.24x |

Output is identical to the scalar renderer within path-tracing noise (~47 dB PSNR
against a scalar reference, at the run-to-run noise floor).

## Build & Run

### Windows (Visual Studio)
```bat
build_windows.bat          :: build pathtracer.exe
build_windows.bat run      :: build, then start the interactive viewer
```
Needs Visual Studio 2019 or 2022 with "Desktop development with C++". The script
finds the compiler itself, and on the first run downloads the official SDL2
development package into `third_party/` (checksum verified).

### Linux
```bash
g++ -O3 -mavx2 -pthread -std=c++17 trace.cpp -o pathtracer -lSDL2
./pathtracer
```

## Command-Line Options
```bash
./pathtracer                         # interactive viewer
./pathtracer demo.json --play        # play a recorded camera path in the window
./pathtracer --bench                 # fixed benchmark: five views, timings and images
./pathtracer --benchmark             # play demo.json in real time and write benchmark_results.json
./pathtracer --offline --samples 128 --resolution 5   # render demo.json to output/frame_NNNNN.png
./pathtracer --help
```

| Option | Meaning |
|---|---|
| `demo.json` or `--demo <file>` | Camera path file to load (default `demo.json`). This is a camera path, not a scene: the world is generated from a seed. |
| `--play` | Play the camera path in the window |
| `--bench` | Fixed benchmark (see below), without a window |
| `--benchmark` | Play the camera path in real time and write `benchmark_results.json` |
| `--offline` | Render the camera path to `output/` as PNG at 30 frames per second, without a window |
| `--samples <n>` | Samples per pixel: offline frames (default 1000), `--bench` (default 32) |
| `--resolution <1-6>` | 144p, 240p, 360p (default), 480p, 720p, 1080p |
| `--threads <n>` | Render threads (default: all) |
| `--seed <n>` | World seed (default 42) |
| `--caustic-quality <1-3>` | 8, 16 or 32 caustic samples (default 3) |
| `--no-caustics`, `--no-volumetrics` | Turn an effect off |

### Fixed benchmark
`--bench` renders five fixed views (lake, shore, underwater, lake bed, aerial) at
a fixed sample count and prints the time, rays per second and samples per second
for each. It writes the images to `output/bench_<view>.png` and the numbers to
`benchmark_fixed.json`. The views, sample count and random sequences are the same
on every machine, so results compare across builds and computers. (`--benchmark`
plays a camera path in real time, so what it renders depends on the machine's
speed.)

```bash
./pathtracer --bench                              # 360p, 32 samples per pixel
./pathtracer --bench --samples 128 --resolution 5 # 720p, 128 samples per pixel
```

Renders are repeatable: random numbers are seeded per pixel, pass and frame, so
the same command gives a byte-identical image on any number of threads. To check
a change against a previous build, keep the old images and compare:

```bash
python compare_images.py output_before output      # PSNR and difference per view
```

### Controls (interactive)
| Keys | Action |
|---|---|
| `W` `A` `S` `D`, `Space`, `Shift`, mouse | Move and look |
| `1` to `6` | Render resolution |
| `Q` / `E` | Window size |
| `R` / `F` | New random world / next seed |
| `T` / `G` | Time of day |
| `F1` | Start or stop recording a camera path |
| `F2` / `F3` | Play the path / benchmark it |
| `F5` / `F6` | Save / load the path |
| Keypad `1` `2` `3` | Toggle caustics, toggle volumetrics, caustic quality |
| `Esc` | Quit |

The water animates while the view is changing and holds still while a still
view accumulates samples, so the image converges. In offline renders the water
follows each frame's time, so its speed does not depend on `--samples`.

### Video Creation
```bash
# Basic video creation (uses output/ directory)
python create_video.py

# Specify custom input/output
python create_video.py -i output -o my_render.mp4

# Adjust frame rate
python create_video.py --fps 60

# Use specific number of CPU cores
python create_video.py -w 8

# Benchmark PPM readers (frames from older builds)
python create_video.py --benchmark
```
Offline frames are PNG; the script also reads the PPM frames older builds wrote.

## Technical Highlights
- Custom Vec3 backed by SSE registers; 8-wide AVX2 sin/cos/exp kernels (no FMA required)
- Caustics sampling, volumetric scattering and value-noise textures vectorized 8-wide
- Volumetric shadow rays marched as 8-wide SIMD packets through the voxel DDA
- Voxel grid traversal (3D DDA); rays that start outside the world are clipped to it
- Stratified sampling of the water surface for caustics

## Requirements
```bash
# For video generation and compare_images.py
pip install -r requirements.txt
```


## License
MIT
