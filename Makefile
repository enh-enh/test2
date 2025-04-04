CFLAG = -O3 -g -Wall -fopenmp -lcudart

all:
	nvcc driver.cc winograd.cu -std=c++17 ${CFLAG} -o winograd

clean:
	rm -f winograd
