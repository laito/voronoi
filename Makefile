voronoi: voronoi.o main.o
	nvcc -o voronoi main.o voronoi.o -g -O3 -I/usr/local/cuda/include  -L./devil/lib  -lm  -lstdc++ -arch=sm_20 -lcudart -L/usr/local/cuda/lib64  -L./devil/lib -I./devil/include

main.o: main.cu
	nvcc -c main.cu -g -O3 -I/usr/local/cuda/include  -L./devil/lib -I./devil/include -lm  -lstdc++ -lcudart -arch=sm_20 -L/usr/local/cuda/lib64   -I./devil/include

voronoi.o: voronoi.cu kernels.o
	nvcc -c voronoi.cu kernels.o -g -O3 -I/usr/local/cuda/include  -L./devil/lib -I./devil/include -lm  -lstdc++ -arch=sm_20 -lcudart -L/usr/local/cuda/lib64  -L./devil/lib -I./devil/include

kernels.o: kernels.cu
	nvcc -c kernels.cu -g -O3 -I/usr/local/cuda/include  -L./devil/lib -I./devil/include -lm  -lstdc++ -arch=sm_20 -lcudart -L/usr/local/cuda/lib64  -L./devil/lib -I./devil/include
clean:
	rm -f *.o *.png  voronoi 
