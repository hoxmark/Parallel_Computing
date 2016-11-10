heat_cuda : heat_cuda.cu
	nvcc -ccbin=g++-4.8 -O3 heat_cuda.cu -o heat_cuda -arch sm_20

clean:
	-rm -f heat_cuda
	-rm -f data/*
