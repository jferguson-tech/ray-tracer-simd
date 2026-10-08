#!/bin/bash
# Prints the SHA-256 of every image the CI renders, one "hash path" per line.
# Run it in the directory that holds output_first/, output_bench/ and
# output_denoised/ (see .github/workflows/build.yml). The result is compared
# with tests/images_linux.sha256 or tests/images_windows.sha256. The two
# platforms round a few math functions differently, so each has its own list.
# After a change that is meant to alter the picture, replace both lists: the
# failed CI step prints the new hashes and keeps the images as an artifact.
export LC_ALL=C
for f in output_first/frame_*.png output_bench/bench_*.png output_denoised/bench_*.png; do
    echo "$(sha256sum "$f" | cut -c1-64) $f"
done | sort -k2
