
#ifndef _RANDOM_MANAGER_
#define _RANDOM_MANAGER_

#include <stdlib.h>
#include <cstdlib>

class Random {

public:
	double general() {
		return (double)rand()/RAND_MAX;
	}
}

#endif

