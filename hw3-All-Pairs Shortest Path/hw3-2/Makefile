CC = gcc
CXX = g++
NVCC = nvcc
HIPCC = hipcc

NVFLAGS = -std=c++11 -O3 -Xptxas="-v" -arch=sm_61 
HIPCCFLAGS = -std=c++11 -O3 --offload-arch=gfx90a

LDFLAGS = -lm
EXES = hw3-2

.PHONY: all clean

all: $(EXES)

clean:
	rm -f $(EXES)

seq: seq.cc
	$(CXX) $(CXXFLAGS) -o $@ $?

hw3-2: hw3-2.cu
	$(NVCC) $(NVFLAGS) $(LDFLAGS) -o $@ $?

hw3-2-amd: hw3-2.hip
	$(HIPCC) $(HIPCCFLAGS) $(LDFLAGS) -o $@ $<

