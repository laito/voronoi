#ifndef _VORONOI_H_
#define _VORONOI_H_

#include <stdio.h>
#include <iostream>
#include <stdlib.h>
#include <complex>
#include <string.h>
#include <float.h>
#include <time.h>
#include <cuda.h>
#include "textures.h"

#define N 128
#define NUM_SEEDS 100

using namespace std;

enum Technique {
	NAIVE,
	JFA
};

class Voronoi {
	public:
		void compute(Technique);

	private:
		void compute_naive();
		void init_random_seeds();
		void init();
		
		int num_seeds;

		int *voronoi;
		int *d_voronoi;

		int4 *seeds;
		int4 *d_seeds;
};

#endif
