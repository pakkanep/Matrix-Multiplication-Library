CXX = g++
CXXFLAGS = -Wall -O3 -std=c++20 -march=native -fopenmp -Isrc/matmul_CPU -MMD -MP
LDFLAGS = -fopenmp

TARGET = main
OBJS = src/main.o src/matmul_CPU/matrix.o
DEPS = $(OBJS:.o=.d)

all: $(TARGET)

$(TARGET): $(OBJS)
	$(CXX) $(OBJS) $(LDFLAGS) -o $(TARGET)

clean:
	rm -f $(OBJS) $(DEPS) $(TARGET)

.PHONY: all clean

-include $(DEPS)