#ifndef _VORONOI_H_
#define _VORONOI_H_

#include <stdio.h>
#include <iostream>
#include <stdlib.h>
#include <complex>
#include <string.h>
#include <IL/il.h>
#include <IL/ilu.h>
#include <float.h>
#include <time.h>
#include <cuda.h>
#include <string.h>
#include "textures.h"

#define NUM_SHAPES 4

using namespace std;

enum Technique {
	NAIVE,
	JFA,
	ONEJFA,
	JFAONE,
	JFATWO,
	ALL
};


class Voronoi {
	public:
		Voronoi(Technique variant_, int size_, int cube_size_, int num_seeds_, bool show_seeds_) : variant(variant_), size(size_), cube_size(cube_size_),  num_seeds(num_seeds_), m_show_seeds(show_seeds_) {}
		void compute();

	private:
		void compute_naive();
		void compute_jfa(Technique);
		void init_random_seeds();
		void allocate_memory();
		void allocate_textures();
		void color_seeds();
		void init();
		void save_slice(int, Technique);
		void save_raw(Technique);
		void save_image(unsigned char *, string, int);
		void show_seeds();
		void show_cube(int, int, int);
		void show_line(int, int, int);
		void show_sphere(int, int, int);
		int get_num_seeds();
		string get_basedir(Technique); 

		int num_seeds;
		int num_cubes;
		int *voronoi;
		int *d_voronoi;
		
		int *valid_cubes;
		int *d_valid_cubes;

		int4 *seeds;
		int4 *d_seeds;
		unsigned char *seed_colors;
		Technique variant;
        int size;
        int num_eeds;
        int cube_size;
        bool m_show_seeds;

};

#endif
