#include "voronoi.h"
#include "kernels.cu"
#include <string.h>

void Voronoi::init() {
    srand(time(NULL));
	// Allocate memory for 3D Space and Seeds
	voronoi = (int *) malloc(sizeof(int *)*size*size*size);
	// Creating a separate array for naive seeds since we have to store seeds and their consituents as seeds
	naive_seeds = (int4 *) malloc(sizeof(int4)*size*size*size);
	seeds = (int4 *) malloc(sizeof(int4)*num_seeds);
	// Create N Random seeds
	init_random_seeds();
}

void Voronoi::init_random_seeds() {
	color_seeds();
	// Assign Seeds to CUBE IDs randomly
	num_cubes = (size / cube_size) * (size / cube_size) * (size / cube_size);
	valid_cubes = (int *) malloc(sizeof(int *)*num_cubes);
	memset(valid_cubes, -1, sizeof(int *)*num_cubes);
	for(int i = 0; i < num_seeds; i++) {
		int rand_cube_id = rand() % num_cubes;
		while(valid_cubes[rand_cube_id] > -1) {
			rand_cube_id = rand() % num_cubes;
		}
		valid_cubes[rand_cube_id] = i+1;
	}
	// Run the kernel to generate seeds
	allocate_memory();
	int num_threads_per_block = 32;
	int num_blocks = num_cubes / num_threads_per_block; 
	generate_seeds<<<num_blocks, num_threads_per_block>>>(d_voronoi, d_valid_cubes, d_seeds, d_naive_seeds, cube_size, size, mode);
	if(m_show_seeds) {
		cudaMemcpy(seeds, d_seeds, sizeof(int4)*num_seeds, cudaMemcpyDeviceToHost);
		cudaMemcpy(naive_seeds, d_naive_seeds, sizeof(int4)*size*size*size, cudaMemcpyDeviceToHost);
	}
}

void Voronoi::color_seeds() {
    seed_colors = (unsigned char *) malloc(sizeof(unsigned char *)*num_seeds*3);
    seed_colors[0] = 0;
    seed_colors[1] = 0;
    seed_colors[2] = 0;
    for(int k = 1; k < num_seeds + 1; k++) {
        unsigned char mr = 255, mg = 255, mb = 255;
        unsigned char r = rand() % 255;
        unsigned char g = rand() % 255;
        unsigned char b = rand() % 255;
        for(int iter = 0; iter < 1; iter++) {
            r = (r+mr)/2;
            g = (g+mg)/2;
            b = (b+mb)/2;
        }
        seed_colors[k*3 + 0] = r;
        seed_colors[k*3 + 1] = g;
        seed_colors[k*3 + 2] = b;
    }
}

void Voronoi::allocate_memory() {
    /* Allocate memory on device */
    cudaMalloc(&d_voronoi, sizeof(int *)*size*size*size);
	cudaMemset(d_voronoi, 0, sizeof(int *)*size*size*size);
    cudaMalloc(&d_valid_cubes, sizeof(int *)*num_cubes);
    cudaMalloc((void **) &d_seeds, sizeof(int4)*num_seeds);
    cudaMalloc((void **) &d_naive_seeds, sizeof(int4)*size*size*size);
	cudaMemset(d_seeds, -1, sizeof(int4)*num_seeds);
	cudaMemset(d_naive_seeds, -1, sizeof(int4)*size*size*size);
    cudaMemcpy(d_valid_cubes, valid_cubes, sizeof(int *)*num_cubes, cudaMemcpyHostToDevice);
}


void Voronoi::allocate_textures() {
    cudaBindTexture(0, tex_seeds, d_seeds, sizeof(int4)*num_seeds);
    cudaBindTexture(0, tex_naive_seeds, d_naive_seeds, sizeof(int4)*size*size*size);
}

void Voronoi::compute_streaming_jfa() {
	printf("Streaming JFA slice-by-slice\n");
	cudaMemcpy(voronoi, d_voronoi, sizeof(int *)*size*size*size, cudaMemcpyDeviceToHost);
	int *ping[size], *pong[size], *plane[size];
	cudaStream_t streams[size];
	for(int i = 0; i < size; i++) {

// 		plane[i] = (int *) malloc(sizeof(int *)*size*size);
//       memset(plane[i], 0, sizeof(int *)*size*size);
		cudaStreamCreateWithFlags(&streams[i], cudaStreamNonBlocking);
		cudaMalloc(&plane[i], sizeof(int *)*size*size);
		int thread_dim = 8;
		int block_dim = cbrt((float)(size*size*size) / (float)(thread_dim*thread_dim*thread_dim));
		dim3 grid(block_dim, block_dim, block_dim);
		dim3 threadBlock(thread_dim, thread_dim, thread_dim);
		plane_generate_kernel<<<grid, threadBlock, 0, streams[i]>>>(d_voronoi, plane[i], i, size);

		/* Serial Implementation for the same */
		/*
		for(int z = 0; z < size; z++) {
            for(int y = 0; y < size; y++) {
                for(int x = 0; x < size; x++) {
                    int value = voronoi[x + y*size + z*size*size];
                    if(value > 0) {
                        int curvalue = plane[i][x + y*size];
                        // We found a seeed, now map it to x-y plane
                        if(curvalue == 0) {
                            plane[i][x + y*size] = value;
                        } else {
                            // Conflict!
                            // Get the closeset seed for this x-y point
                            int4 curseed = seeds[curvalue - 1];
                            int cx = curseed.x - x;
                            int cy = curseed.y - y;
                            int cz = curseed.z - i;
                            int4 otherseed = seeds[value - 1];
                            int ox = otherseed.x - x;
                            int oy = otherseed.y - y;
                            int oz = otherseed.z - i;
                            float cur_distance = cx*cx + cy*cy + cz*cz;
                            float other_distance = ox*ox + oy*oy + oz*oz;
                            if(cur_distance > other_distance) {
                                plane[i][x + y*size] = value;
                            }
                        }
                    }
                }
            }
        }
		*/
	}
	cudaDeviceSynchronize();
	// Run JFA on these planes
	/* Perform ONE JFA */
	int thread_dim = 32;
    int block_dim = sqrt( (size*size) / (thread_dim*thread_dim));
    dim3 grid(block_dim, block_dim);
    dim3 threadBlock(thread_dim, thread_dim);

	for(int k = size/2; k > 0; k = k >> 1) {
		printf("Running kernel streams for k=%d\n", k);
		int thread_dim = 32;
   		int block_dim = sqrt( (size*size) / (thread_dim*thread_dim));
   		dim3 grid(block_dim, block_dim);
   		dim3 threadBlock(thread_dim, thread_dim);
		for(int i = 0; i < size; i++) {
			if(k == size/2) { // First Run
				cudaStreamCreateWithFlags(&streams[i], cudaStreamNonBlocking);
				cudaMalloc(&ping[i], sizeof(int *)*size*size);
            	cudaMalloc(&pong[i], sizeof(int *)*size*size);
				cudaMemcpyAsync(ping[i], plane[i], sizeof(int *)*size*size, cudaMemcpyHostToDevice, streams[i]);
				cudaMemcpyAsync(pong[i], ping[i], sizeof(int *)*size*size, cudaMemcpyDeviceToDevice, streams[i]);
			} else {
				cudaMemcpyAsync(ping[i], pong[i], sizeof(int *)*size*size, cudaMemcpyDeviceToDevice, streams[i]);
			}
       		jfa_voronoi_kernel_planar<<<grid, threadBlock, 0, streams[i]>>>(ping[i], pong[i], k, size, i);
		}
       	cudaDeviceSynchronize();
       	cudaError_t error = cudaDeviceSynchronize();
       	if(error != cudaSuccess) {
           	printf("CUDA error: %s\n", cudaGetErrorString(error));
           	exit(-1);
       	}
   	}
	for(int z = 0; z < size; z++) {
		int *temp = (int *) malloc(sizeof(int *)*size*size);
		cudaStreamSynchronize(streams[z]);
		cudaMemcpyAsync(temp, pong[z], sizeof(int *)*size*size, cudaMemcpyDeviceToHost, streams[z]);
		cudaStreamSynchronize(streams[z]);
		for(int y = 0; y < size; y++) {
			for(int x = 0; x < size; x++) {
				voronoi[x + y*size + z*size*size] = temp[x + y*size];
			}
		}
	}
	show_seeds();
	for(int z = 0; z < size; z++) {
		save_slice(z, STREAMINGJFA);
	}
}

/* Compute Vornonoi via JFA using Ping Pong Buffers */
void Voronoi::compute_jfa(Technique variant) {
	int *temp_voronoi;
	cudaMalloc(&temp_voronoi, sizeof(int *)*size*size*size);
	cudaMemcpy(temp_voronoi, d_voronoi, sizeof(int *)*size*size*size, cudaMemcpyDeviceToDevice);

	if(variant == STREAMINGJFA) {
		compute_streaming_jfa();
		return;
	}
    // Set up the ping pong buffers
    int *ping, *pong;
    cudaMalloc(&ping, sizeof(int *)*size*size*size);
    cudaMalloc(&pong, sizeof(int *)*size*size*size);
    cudaMemcpy(ping, d_voronoi, sizeof(int *)*size*size*size, cudaMemcpyDeviceToDevice);
    cudaMemcpy(pong, ping, sizeof(int *)*size*size*size, cudaMemcpyDeviceToDevice);
	int thread_dim = 8;
	int block_dim = cbrt((float)(size*size*size) / ((float)thread_dim*thread_dim*thread_dim));
    dim3 grid(block_dim, block_dim, block_dim);
    dim3 threadBlock(thread_dim, thread_dim, thread_dim);
	// Extra run for 1-JFA
    if(variant == ONEJFA) {
        jfa_voronoi_kernel<<<grid, threadBlock>>>(d_voronoi, ping, pong, 1, size);
    }
    for(int k = size/2; k > 0; k = k >> 1) {
        jfa_voronoi_kernel<<<grid, threadBlock>>>(d_voronoi, ping, pong, k, size);
        cudaDeviceSynchronize();
        cudaError_t error = cudaDeviceSynchronize();
        if(error != cudaSuccess) {
            printf("CUDA error: %s\n", cudaGetErrorString(error));
            exit(-1);
        }
        cudaMemcpy(ping, pong, sizeof(int *)*size*size*size, cudaMemcpyDeviceToDevice); // Swap the rols of ping pong buffers
    }
	// Extra runs for JFA-1 and JFA-2
	if(variant == JFAONE) {
        jfa_voronoi_kernel<<<grid, threadBlock>>>(d_voronoi, ping, pong, 1, size);
    } else if (variant == JFATWO) {
        jfa_voronoi_kernel<<<grid, threadBlock>>>(d_voronoi, ping, pong, 2, size);
        jfa_voronoi_kernel<<<grid, threadBlock>>>(d_voronoi, ping, pong, 1, size);
    }
	cudaMemcpy(voronoi, pong, sizeof(int *)*size*size*size, cudaMemcpyDeviceToHost);
	// Whether the raw/images should have seeds printed on them or not
	if(m_show_seeds) {
    	show_seeds();
	}
    /* Save to Dist */
    //save_raw(variant);
    for(int z = 0; z < size; z++) {
        save_slice(z, variant);
    }
	/* Reset Vornoi Space */
	cudaMemcpy(d_voronoi, temp_voronoi, sizeof(int *)*size*size*size, cudaMemcpyDeviceToDevice);
}

/* Seed Printing Functions */
/* These functions when called will color the respective locations of the seed shapes in the final 3D volume */

void Voronoi::show_cube(int x, int y, int z) {
    int side = cube_size;
    for(int cz = z; cz < z + side; cz++) {
        for(int cy = y; cy < y + side; cy++) {
            for(int cx = x; cx < x + side; cx++) {
                if(cx >= 0 && cy >= 0 && cz >= 0 && cy) {
                    voronoi[cx + cy*size + cz*size*size] = -1;
                }
            }
        }
    }
}

void Voronoi::show_line(int x, int y, int z) {
    int l = cube_size;
    int limit = x + l;
    while(x < limit) {
        x += 1;
        y += 1;
        z += 1;
        if(x < size && y < size && z < size) {
            voronoi[x + y*size + z*size*size] = -1;
        }
    }
}

void Voronoi::show_sphere(int x, int y, int z) {
    int r = (cube_size / 2) - 1;
    for(int _x = -r; _x <= r; _x++) {
        for(int _y = -r; _y <= r; _y++) {
            for(int _z = -r; _z <= r; _z++) {
                int posx = _x + x;
                int posy = _y + y;
                int posz = _z + z;
                if(sqrt(_x*_x + _y*_y + _z*_z) <= r) {
                    voronoi[posx + posy*size+ posz*size*size] = -1;
                }
            }
        }
    }
}


void Voronoi::show_seeds() {
    int line_count = 0;
    int point_count = 0;
    int sphere_count = 0;
    int cube_count = 0;
    for(int i = 0; i < num_seeds; i++) {
        int shape = seeds[i].w;
        int x = seeds[i].x;
        int y = seeds[i].y;
        int z = seeds[i].z;
        switch(shape) {
            case 0: /* POINT */
                point_count++;
                voronoi[x + y*size+ z*size*size] = -1;
                break;
            case 1: /* CUBOID */
                cube_count++;
                show_cube(x, y, z);
                break;
            case 2:
                sphere_count++;
                show_sphere(x, y, z);
                break;
            case 3:
                line_count++;
                show_line(x, y, z);
                break;
        }
    }
    //printf("Lines: %d, Cubes: %d, Points: %d, Spheres: %d\n", line_count, cube_count, point_count, sphere_count);
}


/* Runt he naive algorithm - Go through each seed iteratiely to find out the closest seed */
void Voronoi::compute_naive() {
	int *temp_voronoi;
	cudaMalloc(&temp_voronoi, sizeof(int *)*size*size*size);
	cudaMemcpy(temp_voronoi, d_voronoi, sizeof(int *)*size*size*size, cudaMemcpyDeviceToDevice);
	int thread_dim = 8;
    int block_dim = cbrt((float)(size*size*size) / (float)(thread_dim*thread_dim*thread_dim));
    dim3 grid(block_dim, block_dim, block_dim);
    dim3 threadBlock(thread_dim, thread_dim, thread_dim);
	naive_voronoi_kernel<<<grid, threadBlock>>>(d_voronoi, size*size*size, size);
	cudaMemcpy(voronoi, d_voronoi, sizeof(int *)*size*size*size, cudaMemcpyDeviceToHost);
	show_seeds();
	for(int z = 0; z < size; z++) {
		save_slice(z, NAIVE);
	}
	cudaMemcpy(d_voronoi, temp_voronoi, sizeof(int *)*size*size*size, cudaMemcpyDeviceToDevice);
}

string Voronoi::get_basedir(Technique variant) {
    string basedir = "output_images/";
    switch(variant) {
        case JFA:
            basedir += "jfa";
            break;
        case ONEJFA:
            basedir += "1jfa";
            break;
        case JFAONE:
            basedir += "jfa1";
            break;
        case JFATWO:
            basedir += "jfa2";
            break;
		case NAIVE:
			basedir += "naive";
			break;
		case STREAMINGJFA:
			basedir += "streaming_jfa";
			break;
        default:
            break;
    }
    return basedir;
}
void Voronoi::save_raw(Technique variant) {
    string basedir = get_basedir(variant);
    FILE *fp = fopen((basedir+"/cube.raw").c_str(), "wb");
	int r_counter = 0;
    for(int z = 0; z < size; z++) {
        for(int y = 0; y < size; y++) {
            for(int x = 0; x < size; x++) {
                int val = voronoi[x + y*size + z*size*size] - 1;
                unsigned char r;
                if(val < 0) {
                    r = 0;
                } else {
                    r = r_counter++;//seed_colors[val*3 + 0];
                }
                fwrite(&r, sizeof(unsigned char), 1, fp);
            }
        }
    }
    fclose(fp);
}

void Voronoi::save_slice(int z, Technique variant) {
    string basedir = get_basedir(variant);
    unsigned char *bitmap = (unsigned char *) malloc(sizeof(unsigned char *)*size*size*3);
    for(int y = 0; y < size; y++) {
        for(int x = 0; x < size; x++) {
            int val = voronoi[x + y*size + z*size*size];
            if(val == -1) {
                 bitmap[(x + y*size)*3 + 0] = 0;
                 bitmap[(x + y*size)*3 + 1] = 0;
                 bitmap[(x + y*size)*3 + 2] = 0;
            } else {
                bitmap[(x + y*size)*3 + 0] = seed_colors[val*3  + 0];
                bitmap[(x + y*size)*3 + 1] = seed_colors[val*3 + 1];
                bitmap[(x + y*size)*3 + 2] = seed_colors[val*3 + 2];
            }
        }
    }
    save_image(bitmap, basedir, z);
}

void Voronoi::save_image(unsigned char * bitmap, string basedir, int counter) {
    ilInit();
    ILuint imageID = ilGenImage();
    ilBindImage(imageID);
    ilTexImage(size, size, 1, 3, IL_RGB, IL_UNSIGNED_BYTE, bitmap);
    ilEnable(IL_FILE_OVERWRITE);
    std::ostringstream stringStream;
    stringStream << basedir+"/output_" << counter << ".png";
    string filename = stringStream.str();
    ilSave(IL_PNG, filename.c_str());
}


void Voronoi::compute() {
	init();
	allocate_textures();
	if(mode == NAIVE) {
		compute_naive();
	} else if (mode == ALL) {
		compute_naive();
		compute_jfa(JFA);
		compute_jfa(JFAONE);
		compute_jfa(ONEJFA);
		compute_jfa(JFATWO);
		compute_jfa(STREAMINGJFA);
	} else {
//		compute_naive();
		compute_jfa(mode);
	}
}
