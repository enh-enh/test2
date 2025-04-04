CFLAG = -O3 -g -Wall -fopenmp -lcudart

all:
	g++ driver.cc winograd.cc -std=c++17 ${CFLAG} -o winograd

clean:
	rm -f winograd
