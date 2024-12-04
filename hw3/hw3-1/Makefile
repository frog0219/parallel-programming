CC = gcc
CXX = g++
CFLAGS = -lm -O3
hw3-1 : CC = mpicc
hw3-1 : CFLAGS += -fopenmp
CXXFLAGS = $(CFLAGS)
TARGETS = hw3-1

.PHONY: all
all: $(TARGETS)

.PHONY: clean
clean:
	rm -f $(TARGETS) $(TARGETS:=.o)
