CFLAG = -O3 -g -Wall -fopenmp -lcudart
CFLAG2 = -O3 -g

all:
	nvcc driver.cc winograd.cu -std=c++17 ${CFLAG2} -o winograd

clean:
	rm -f winograd
