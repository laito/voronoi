#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <curand.h>
#include <curand_kernel.h>
#include <math_functions.h>
#include "voronoi.h"


__device__ void draw_cube(int *, int4 *, int, int, int, int, int, int, Technique);
__device__ void draw_sphere(int *, int4 *, int, int, int, int, int, int, Technique);
__device__ void draw_line(int *, int4 *, int, int, int, int, int, int, Technique);


/* Kernel to generate generalized sites */
__global__ void generate_seeds(int *voronoi, int *valid_cubes, int4 *seeds, int cell_size, int size, Technique variant) {
	int id = blockDim.x*blockIdx.x + threadIdx.x;
	int num_cells = size/cell_size; /* sizeumber of cells along each axis */
	int y_i = (id%num_cells)*cell_size;
	int level = id/(num_cells * num_cells);
	int z_i = (id/(num_cells * num_cells))*cell_size;
	int x_i = ((id - num_cells*num_cells*level)/num_cells)*cell_size;
	/* Generate Seed */
	int seed_counter = valid_cubes[id];
	if (seed_counter > -1) {
		//voronoi[(x_i) + (y_i)*size + (z_i)*size*size] = id + 1;
		voronoi[(x_i) + (y_i)*size + (z_i)*size*size] = seed_counter;

		int finalx = x_i;
		int finaly = y_i;
		int finalz = z_i;

		int seed = clock64();
		curandState s;
		curand_init(seed, 0, 0, &s);
		int shape = curand_uniform(&s) * 4;
		switch(shape) {
			case 0: /* Point */
				break;
			case 1: /* Cuboid */
				draw_cube(voronoi, seeds, 16, x_i, y_i, z_i, seed_counter, size, variant);
				break;
			case 2: /* Sphere */
				finalx += 8;
				finaly += 8;
				finalz += 8;
				draw_sphere(voronoi, seeds, 7, x_i + 8, y_i + 8, z_i + 8, seed_counter, size, variant);
				break;
			case 3: /* Line */
				draw_line(voronoi, seeds, 15, x_i, y_i, z_i, seed_counter, size, variant);
				break;
		}
		// We only need NAIVE for Error calculation, so we don't need the information of shape of seed
		// For naive, we'll only need the right color for that seed
		if(variant == NAIVE) {
			shape = seed_counter;
			seed_counter = x_i + y_i*size + z_i*size*size;	
		}
		seeds[seed_counter].w = shape;
	 	seeds[seed_counter].x = finalx;
        seeds[seed_counter].y = finaly;
        seeds[seed_counter].z = finalz;
	}
}

__device__ void draw_line(int *voronoi, int4* seeds, int l, int x, int y, int z, int color, int size, Technique variant) {
	int limit = x + l;
	while(x < limit) {
		x += 1;
		y += 1;
		z += 1;
		if(x < size && y < size && z < size) {
			voronoi[x + y*size + z*size*size] = color;
			if(variant == NAIVE) {
				seeds[x + y*size + z*size*size].x = x;
				seeds[x + y*size + z*size*size].y = y;
				seeds[x + y*size + z*size*size].z = z;
				seeds[x + y*size + z*size*size].w = color;
			}
		}
	}
}

__device__ void draw_sphere(int *voronoi, int4* seeds, int r, int posx, int posy, int posz, int color, int size, Technique variant) {
	for(int _x = -r; _x <= r; _x++) {
        for(int _y = -r; _y <= r; _y++) {
            for(int _z = -r; _z <= r; _z++) {
                int x = _x + posx;
                int y = _y + posy;
                int z = _z + posz;
                if(sqrtf(_x*_x + _y*_y + _z*_z) <= r) {
                    voronoi[x + y*size + z*size*size] = color;
					if(variant == NAIVE) {
                		seeds[x + y*size + z*size*size].x = x;
                		seeds[x + y*size + z*size*size].y = y;
                		seeds[x + y*size + z*size*size].z = z;
                		seeds[x + y*size + z*size*size].w = color;
            		}
                }
            }
        }
    }
}

__device__ void draw_cube(int *voronoi, int4* seeds, int side, int cx, int cy, int cz, int color, int size, Technique variant) {
	for(int z = cz; z < cz + side; z++) {
		for(int y = cy; y < cy + side; y++) {
			for(int x = cx; x < cx + side; x++) {
				if(x >= 0 && y >= 0 && z >= 0 && y < size && x < size && z < size) {
					voronoi[x + y*size + z*size*size] = color;
					if(variant == NAIVE) {
                		seeds[x + y*size + z*size*size].x = x;
                		seeds[x + y*size + z*size*size].y = y;
                		seeds[x + y*size + z*size*size].z = z;
                		seeds[x + y*size + z*size*size].w = color;
            		}
				}
			}
		}
	}
}

__global__ void jfa_voronoi_kernel_planar(int *ping, int *pong, int k, int size, int z) {
	int x = threadIdx.x + blockIdx.x*blockDim.x;
	int y = threadIdx.y + blockIdx.y*blockDim.y;

	int seed = ping[x + y*size];
	if(seed < 0) {
		return;
	}

	int values[3]; values[0] = -k; values[1] = 0; values[2] = k;
	for(int counter = 0; counter < 9; counter++) {
		int _x = values[counter / 3];
		int _y = values[counter % 3];
		int posx = x + _x;
		int posy = y + _y;
		if(posx >= 0 && posy >= 0 && posx < size && posy < size) {
			int index = posx + posy*size;
			int value = pong[index];
			if(value <= 0) {
				pong[index] = seed;
			} else {
				/* Calculate distance */
                int4 curseed = tex1Dfetch(tex_seeds, seed - 1);
                int cx = curseed.x - posx;
                int cy = curseed.y - posy;
                int cz = curseed.z - z;

                int4 otherseed = tex1Dfetch(tex_seeds, value - 1);
                int ox = otherseed.x - posx;
                int oy = otherseed.y - posy;
                int oz = otherseed.z - z;

                float curDistance = cx*cx + cy*cy + cz*cz;
                float otherDistance = ox*ox + oy*oy + oz*oz;

                if(curDistance < otherDistance) {
                    pong[index] = seed;
                }
			}
		}
	}
}

/* JFA Kernel */
__global__ void jfa_voronoi_kernel(int *voronoi, int *ping, int *pong, int k, int size) {
	int x = threadIdx.x + blockIdx.x*blockDim.x;
    int y = threadIdx.y + blockIdx.y*blockDim.y;
    int z = threadIdx.z + blockIdx.z*blockDim.z;
		
	int seed = ping[x + y*size + z*size*size];
	if(seed <= 0) {
		return;
	}
	
	int values[3]; values[0] = -k; values[1] = 0; values[2] = k;
	for(int counter = 0; counter < 27; counter++ ) {
		int _z = values[counter % 3];
		int _x = values[counter / 9];
		int _y = values[(counter - 9*(counter/9)) / 3];
			
		int posx = x + _x;
		int posy = y + _y;
		int posz = z + _z;

		if(posx >= 0 && posx < size && posy >= 0 && posy < size) {
			int index = posx + size*posy + size*size*posz;
			int value = pong[index];
			if(value <= 0) {
				pong[index] = seed;
			} else {
				/* Calculate distance */ 
				int4 curseed = tex1Dfetch(tex_seeds, seed - 1);
				int cx = curseed.x - posx;
				int cy = curseed.y - posy;
				int cz = curseed.z - posz;

				int4 otherseed = tex1Dfetch(tex_seeds, value - 1);
				int ox = otherseed.x - posx;
				int oy = otherseed.y - posy;
				int oz = otherseed.z - posz;
				
				float curDistance = cx*cx + cy*cy + cz*cz;
				float otherDistance = ox*ox + oy*oy + oz*oz;
				
				if(curDistance < otherDistance) {
					pong[index] = seed;
				}
			}
		}
	} 
}


/* sizeAIVE Kernel */

__global__ void naive_voronoi_kernel(int *voronoi, int num_seeds, int size) {

	int x = threadIdx.x + blockIdx.x*blockDim.x;
	int y = threadIdx.y + blockIdx.y*blockDim.y;
	int z = threadIdx.z + blockIdx.z*blockDim.z;

	if(x < size && y < size && z < size) {
		int k;
		int _x, _y, _z, dx, dy, dz;
		float dist2;
		float min_dist = FLT_MAX;
		int mink = 0;
		for(k = 0; k < num_seeds; k++) {
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
				mink = curseed.w;
			}
		}
		voronoi[(x + y*size + z*size*size)] = mink + 1;
	}
}
