#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include "voronoi.h"

using namespace std;

int main() {
	Voronoi *controller = new Voronoi();
	controller->compute(NAIVE);
	return 0;
}


