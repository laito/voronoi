#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include "voronoi.h"

using namespace std;

int main(int argc, char *arvgv[]) {
	// Set up defaults
	//int size = 128;
	int size = 128;
	int cube_size = 16;
	int num_seeds = 100;
	bool show_seeds = true;
	// Set up and run the controller with arguments
	//Voronoi *controller = new Voronoi(STREAMINGJFA, size, cube_size, num_seeds, show_seeds);
	Voronoi *controller = new Voronoi(STREAMINGJFA, size, cube_size, num_seeds, show_seeds);
	controller->compute();
	return 0;
}


