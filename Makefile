CXX = g++
CXXFLAGS = -Wall -O3 -march=native -fopenmp -std=c++20
LDFLAGS = -fopenmp

TARGET = main
OBJS = src/main.o src/matmul_CPU/matrix.o

all: $(TARGET)

$(TARGET): $(OBJS)
	$(CXX) $(OBJS) $(LDFLAGS) -o $(TARGET)

src/main.o src/matmul_CPU/matrix.o: src/matmul_CPU/matrix.h

clean:
	rm -f $(OBJS) $(TARGET)

.PHONY: all clean