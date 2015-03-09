#include "voronoi.h"
#include "kernels.cu"

void Voronoi::init() {
    srand(time(NULL));
	// Initialize voronoi 3D vector
	voronoi = (int *) malloc(sizeof(int *)*N*N*N);
	seeds = (int4 *) malloc(sizeof(int4)*NUM_SEEDS);
	// Create N Random seeds
	init_random_seeds();
}

void Voronoi::init_random_seeds() {
	for(int i = 0; i < NUM_SEEDS; i++) {
		// Generate random location for seed
		seeds[i].x = rand() % N;
		seeds[i].y = rand() % N;
		seeds[i].z = rand() % N;
	}
}

void Voronoi::compute_naive() {
	cudaMalloc(&d_voronoi, sizeof(int *)*N*N*N);
	cudaMalloc((void **) &d_seeds, sizeof(int4)*NUM_SEEDS);
	
	cudaMemcpy(d_voronoi, voronoi, sizeof(int *)*N*N*N, cudaMemcpyHostToDevice);
	cudaMemcpy(d_seeds, seeds, sizeof(int4)*NUM_SEEDS, cudaMemcpyHostToDevice);

    cudaBindTexture(0, tex_seeds, d_seeds, sizeof(int4)*NUM_SEEDS);

	dim3 grid(16, 16, 16);
	dim3 threadBlock(10, 10, 10);

	naive_voronoi_kernel<<<grid, threadBlock>>>(d_voronoi);

	cudaMemcpy(voronoi, d_voronoi, sizeof(int *)*N*N*N, cudaMemcpyDeviceToHost);
}

void Voronoi::compute(Technique technique) {
	init();
	switch(technique) {
		case NAIVE:
			compute_naive();
			break;
		default:
			break;
	}
}
