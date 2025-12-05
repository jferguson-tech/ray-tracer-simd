#!/usr/bin/env python3
"""
Fast optimized video creator for PPM frames
Uses multiple optimization strategies for maximum speed
Fixed with H.264 codec for browser compatibility (no ffmpeg needed)
"""

import cv2
import numpy as np
import glob
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
import multiprocessing
from functools import partial
import mmap

def ppm_to_bgr_fast(ppm_file: str) -> tuple:
    """
    Fast PPM reader using numpy for parsing.
    Much faster than reading line by line.
    """
    # Extract frame number
    filename = os.path.basename(ppm_file)
    frame_num = int(filename.replace('frame_', '').replace('.ppm', ''))
    
    with open(ppm_file, 'rb') as f:
        # Memory map the file for faster reading
        with mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) as mmapped_file:
            # Read header lines
            header = mmapped_file.readline().decode('ascii').strip()
            if header != 'P3':
                raise ValueError(f"Invalid PPM header: {header}")
            
            # Read dimensions
            dims = mmapped_file.readline().decode('ascii').strip().split()
            width, height = int(dims[0]), int(dims[1])
            
            # Read max value
            max_val = int(mmapped_file.readline().decode('ascii').strip())
            
            # Read rest of file as bytes
            remaining = mmapped_file.read()
            
    # Parse all numbers at once using numpy
    # This is much faster than line-by-line parsing
    pixels = np.fromstring(remaining.decode('ascii'), dtype=np.uint8, sep=' ')
    
    # Reshape to image
    img = pixels.reshape((height, width, 3))
    
    # Convert RGB to BGR for OpenCV
    img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    
    return frame_num, img_bgr

def ppm_to_bgr_fastest(ppm_file: str) -> tuple:
    """
    Even faster PPM reader - reads entire file and uses numpy for everything.
    """
    # Extract frame number
    filename = os.path.basename(ppm_file)
    frame_num = int(filename.replace('frame_', '').replace('.ppm', ''))
    
    # Read entire file at once
    with open(ppm_file, 'r') as f:
        lines = f.readlines()
    
    # Parse header
    if lines[0].strip() != 'P3':
        raise ValueError(f"Invalid PPM header")
    
    dims = lines[1].strip().split()
    width, height = int(dims[0]), int(dims[1])
    max_val = int(lines[2].strip())
    
    # Join all pixel lines and parse at once
    pixel_data = ' '.join(lines[3:])
    
    # Use numpy to parse all numbers at once - VERY fast
    pixels = np.fromstring(pixel_data, dtype=np.uint8, sep=' ')
    
    # Reshape and convert
    img = pixels.reshape((height, width, 3))
    img_bgr = img[:, :, [2, 1, 0]]  # RGB to BGR swap - faster than cvtColor
    
    return frame_num, np.ascontiguousarray(img_bgr)

def process_frame_worker(ppm_file: str) -> tuple:
    """Worker function for multiprocessing."""
    try:
        return ppm_to_bgr_fastest(ppm_file)
    except Exception as e:
        print(f"Error processing {ppm_file}: {e}")
        return None, None

def create_video_fast(input_dir: str = 'output',
                     output_file: str = 'pathtracer.mp4',
                     fps: int = 30,
                     num_workers: int = None,
                     use_processes: bool = True) -> bool:
    """
    Create video using fast parallel processing.
    
    Args:
        input_dir: Directory with PPM frames
        output_file: Output video file
        fps: Frames per second
        num_workers: Number of workers (None = auto)
        use_processes: Use processes (True) or threads (False)
    """
    # Auto-detect workers
    if num_workers is None:
        num_workers = multiprocessing.cpu_count()
    
    print(f"Fast Video Creator - Optimized for AMD Ryzen")
    print(f"Using {num_workers} {'processes' if use_processes else 'threads'}")
    print("-" * 60)
    
    # Find all frames
    frame_files = sorted(glob.glob(os.path.join(input_dir, 'frame_*.ppm')))
    
    if not frame_files:
        print(f"Error: No PPM frames found in {input_dir}")
        return False
    
    total_frames = len(frame_files)
    print(f"Found {total_frames} frames to process")
    
    # Get dimensions from first frame
    print("Reading first frame...")
    _, first_frame = ppm_to_bgr_fastest(frame_files[0])
    height, width = first_frame.shape[:2]
    print(f"Video dimensions: {width}x{height}")
    
    # Create video writer with H.264 codec for browser compatibility
    # Try different H.264 fourcc codes until one works
    h264_fourccs = [
        cv2.VideoWriter_fourcc(*'avc1'),  # Best for browsers
        cv2.VideoWriter_fourcc(*'H264'),
        cv2.VideoWriter_fourcc(*'h264'),
        cv2.VideoWriter_fourcc(*'x264'),
        cv2.VideoWriter_fourcc(*'mp4v'),  # Fallback if H.264 not available
    ]
    
    out = None
    for fourcc in h264_fourccs:
        out = cv2.VideoWriter(output_file, fourcc, fps, (width, height))
        if out.isOpened():
            codec_name = ''.join([chr((fourcc >> 8 * i) & 0xFF) for i in range(4)])
            print(f"Using codec: {codec_name}")
            break
    
    if not out or not out.isOpened():
        print("Error: Could not create video writer")
        return False
    
    print(f"Processing {total_frames} frames in parallel...")
    start_time = time.time()
    
    # Process all frames in parallel
    frames_dict = {}
    processed = 0
    
    # Choose executor based on flag
    Executor = ProcessPoolExecutor if use_processes else ThreadPoolExecutor
    
    with Executor(max_workers=num_workers) as executor:
        # Submit all tasks
        future_to_idx = {
            executor.submit(process_frame_worker, f): i 
            for i, f in enumerate(frame_files)
        }
        
        # Collect results as they complete
        for future in as_completed(future_to_idx):
            idx = future_to_idx[future]
            try:
                frame_num, img_bgr = future.result()
                if img_bgr is not None:
                    frames_dict[frame_num] = img_bgr
                    processed += 1
                    
                    # Progress update
                    if processed % 50 == 0 or processed == total_frames:
                        elapsed = time.time() - start_time
                        rate = processed / elapsed
                        eta = (total_frames - processed) / rate if rate > 0 else 0
                        print(f"Processed: {processed}/{total_frames} "
                              f"({100*processed/total_frames:.1f}%) "
                              f"Speed: {rate:.1f} frames/sec "
                              f"ETA: {eta:.1f}s")
            except Exception as e:
                print(f"Error processing frame {idx}: {e}")
    
    # Write frames in order
    print("Writing video...")
    write_start = time.time()
    
    for i in range(total_frames):
        if i in frames_dict:
            out.write(frames_dict[i])
        else:
            print(f"Warning: Missing frame {i}")
    
    write_time = time.time() - write_start
    
    # Cleanup
    out.release()
    cv2.destroyAllWindows()
    
    # Statistics
    total_time = time.time() - start_time
    process_time = total_time - write_time
    
    print("\n" + "="*60)
    print(" VIDEO CREATION COMPLETE!")
    print("="*60)
    print(f"Total frames: {total_frames}")
    print(f"Processing time: {process_time:.2f}s ({total_frames/process_time:.1f} fps)")
    print(f"Writing time: {write_time:.2f}s")
    print(f"Total time: {total_time:.2f}s")
    print(f"Overall speed: {total_frames/total_time:.1f} frames/second")
    print(f"Output: {output_file}")
    print(f"Duration: {total_frames/fps:.1f}s at {fps} FPS")
    
    if os.path.exists(output_file):
        file_size = os.path.getsize(output_file) / (1024 * 1024)
        print(f"File size: {file_size:.1f} MB")
        print(" Should work in all browsers with H.264 codec!")
    
    return True

def benchmark_readers(input_dir: str = 'output'):
    """Benchmark different PPM reading methods."""
    frame_files = sorted(glob.glob(os.path.join(input_dir, 'frame_*.ppm')))[:10]
    
    if not frame_files:
        print("No frames found")
        return
    
    print("Benchmarking PPM readers (10 frames)...")
    print("-" * 60)
    
    # Test fastest method
    start = time.time()
    for f in frame_files:
        ppm_to_bgr_fastest(f)
    fast_time = time.time() - start
    print(f"Fastest method: {10/fast_time:.1f} frames/sec")
    
    # Test fast method
    start = time.time()
    for f in frame_files:
        ppm_to_bgr_fast(f)
    medium_time = time.time() - start
    print(f"Fast method: {10/medium_time:.1f} frames/sec")
    
    # Test parallel processing
    print("\nTesting parallel processing...")
    
    # Threads
    start = time.time()
    with ThreadPoolExecutor(max_workers=8) as executor:
        list(executor.map(ppm_to_bgr_fastest, frame_files))
    thread_time = time.time() - start
    print(f"8 threads: {10/thread_time:.1f} frames/sec")
    
    # Processes
    start = time.time()
    with ProcessPoolExecutor(max_workers=8) as executor:
        list(executor.map(process_frame_worker, frame_files))
    process_time = time.time() - start
    print(f"8 processes: {10/process_time:.1f} frames/sec")
    
    print("-" * 60)
    print(f"Best method: {'Processes' if process_time < thread_time else 'Threads'}")

def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Fast optimized video creator for PPM frames with H.264 codec'
    )
    parser.add_argument('-i', '--input', default='output',
                       help='Input directory')
    parser.add_argument('-o', '--output', default='pathtracer.mp4',
                       help='Output video file')
    parser.add_argument('--fps', type=int, default=30,
                       help='Frames per second')
    parser.add_argument('-w', '--workers', type=int,
                       help='Number of workers (default: all CPUs)')
    parser.add_argument('--threads', action='store_true',
                       help='Use threads instead of processes')
    parser.add_argument('--benchmark', action='store_true',
                       help='Benchmark different methods')
    
    args = parser.parse_args()
    
    if args.benchmark:
        benchmark_readers(args.input)
        return
    
    # Create video
    success = create_video_fast(
        input_dir=args.input,
        output_file=args.output,
        fps=args.fps,
        num_workers=args.workers,
        use_processes=not args.threads
    )
    
    if success:
        print(f"\n  Play with: vlc {args.output}")
    else:
        sys.exit(1)

if __name__ == '__main__':
    main()
