#!/bin/bash
g++ -O3 -march=native -mavx2 -pthread -std=c++17 trace.cpp -o pathtracer -lSDL2 && ./pathtracer
