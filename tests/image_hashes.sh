#!/bin/bash
# Prints the SHA-256 of every image the CI renders, one "hash path" per line.
# Run it in the directory that holds output_first/, output_bench/ and
# output_denoised/ (see .github/workflows/build.yml). The result is compared
# with tests/images.sha256; after a change that is meant to alter the picture,
# write the new output to that file.
export LC_ALL=C
for f in output_first/frame_*.png output_bench/bench_*.png output_denoised/bench_*.png; do
    echo "$(sha256sum "$f" | cut -c1-64) $f"
done | sort -k2
