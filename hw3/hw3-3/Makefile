CC = gcc
CXX = g++
NVCC = nvcc
HIPCC = hipcc

CXXFLAGS = -O3
NVFLAGS = -std=c++11 -O3 -Xptxas="-v" -arch=sm_61 
HIPCCFLAGS = -std=c++11 -O3 --offload-arch=gfx90a

LDFLAGS = -lm
EXES = hw3-3

hw3-3: NVFLAGS += -Xcompiler="-fopenmp"
hw3-3-amd: HIPCCFLAGS += -fopenmp

.PHONY: all clean

all: $(EXES)

clean:
	rm -f $(EXES)

seq: seq.cc
	$(CXX) $(CXXFLAGS) -o $@ $?

hw3-3: hw3-3.cu
	$(NVCC) $(NVFLAGS) $(LDFLAGS) -o $@ $?

hw3-3-amd: hw3-3.hip
	$(HIPCC) $(HIPCCFLAGS) $(LDFLAGS) -o $@ $<
