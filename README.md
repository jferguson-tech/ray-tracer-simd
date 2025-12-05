# Path Tracer

High-performance C++ path tracer with SIMD acceleration.

![Path Tracer Demo](pathtracer.gif)

## Features
- AVX/SIMD optimization for 4x+ speedup
- OpenMP multi-threading
- Multiple materials (diffuse, metal, dielectric)
- JSON scene description

## Build & Run
```bash
# Build the pathtracer
g++ -O3 -fopenmp -mavx2 trace.cpp -o pathtracer

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
- Custom Vec3 with SIMD operations
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
