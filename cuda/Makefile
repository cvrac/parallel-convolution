convolute:	convolute.o convolute_Filter.o
		nvcc -o convolute convolute.o convolute_Filter.o 

convolute.o: convolute.cpp
		g++ -c convolute.cpp

convolute_Filter.o: convolute_Filter.cu
		nvcc -c convolute_Filter.cu

count:
		wc -l *.h *.cpp *.cu

clean:
		rm -rf *.o convolute *.dat script.sh.*

