#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <math_functions.h>
#include "voronoi.h"
#include "textures.h"

__global__ void naive_voronoi_kernel(int *voronoi) {

	int x = threadIdx.x + blockIdx.x*blockDim.x;
	int y = threadIdx.y + blockIdx.y*blockDim.y;
	int z = threadIdx.z + blockIdx.z*blockDim.z;

	if(x < N && y < N && z < N) {
		int k;
		int _x, _y, _z, dx, dy, dz;
		float dist2;
		float min_dist = FLT_MAX;
		int mink = 0;
	
		for(k = 0; k < NUM_SEEDS; k++) {
			int4 curseed = tex1Dfetch(tex_seeds, k);
			_x = curseed.x;
			_y = curseed.y;
			_z = curseed.z;
			dx = _x - x;
			dy = _y - y;
			dz = _z - z;
			dist2 = dx*dx + dy*dy + dz*dz;
			if(min_dist > dist2) {
				min_dist = dist2;
				mink = k;
			}
		}

		voronoi[(x + y*N + z*N*N)] = mink;
	}
}
