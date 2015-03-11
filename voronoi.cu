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

	printf("Back to CPU\n");

	int lastval = -1;
	for(int z = 0; z < N; z++) {
		for(int y = 0; y < N; y++) {
			for(int x = 0; x < N; x++) {
				
				int val = voronoi[x + y*N + z*N*N];
				printf(" %d ", val);
				int nextval = voronoi[x + y*N + z*N*N + 1];
				if(lastval != -1 && (val != lastval || val != nextval)) {
					voronoi[x + y*N + z*N*N] = 1;
				} else {
					voronoi[x + y*N + z*N*N] = 0;
				}
				lastval = val;
			}
			printf("\n");
		}
	}

	printf("\n\n");
	for(int z = 0; z < 1; z++) {
        for(int y = 0; y < N; y++) {
            for(int x = 0; x < N; x++) {
				printf("%d", voronoi[x + y*N + z*N*N]);
            }
			printf("\n");
        }
    }
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
