# Path Tracer

High-performance C++ path tracer with SIMD acceleration.

![Path Tracer Demo](pathtracer.gif)

## Features
- AVX2 SIMD acceleration: measured 1.9-2.2x whole-frame speedup (3-5x in the vectorized subsystems)
- 8-wide packet ray marching for volumetric shadow rays, vectorized caustics/noise kernels, SSE-backed Vec3
- Multi-threaded tile renderer (std::thread)
- Multiple materials (diffuse, metal, dielectric)
- JSON demo/camera path playback and offline rendering

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
```bash
# Build the pathtracer
g++ -O3 -mavx2 -std=c++17 trace.cpp -o pathtracer -lSDL2

# Run with default scene (built into the code)
./pathtracer

# Run with custom JSON scene
./pathtracer my_scene.json

# Create video from rendered frames
python create_video.py
```

## Command-Line Options

### Pathtracer
```bash
# Default scene (no arguments)
./pathtracer

# Custom scene file
./pathtracer demo.json
```

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

# Benchmark PPM readers
python create_video.py --benchmark
```

## Technical Highlights
- Custom Vec3 backed by SSE registers; 8-wide AVX2 sin/cos/exp kernels (no FMA required)
- Caustics sampling, volumetric scattering and value-noise textures vectorized 8-wide
- Volumetric shadow rays marched as 8-wide SIMD packets through the voxel DDA
- Physically-based BRDF
- Stratified sampling
- BVH acceleration (planned)

## Requirements
```bash
# For video generation
pip install opencv-python numpy
```


## License
MIT
