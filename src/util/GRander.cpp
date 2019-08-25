#include "samp_set.hpp"
#include "GRander.hpp"
using namespace Grusoft;

extern "C" uint64_t xoroshiro_next(void);

uint64_t GRander::RandRersResrResdra() {  // Combined period = 2^116.23
	int alg = 2;
	switch (alg) {
	case 0:
		return pcg32_random_r(&rng_neil);		//32-bit unsigned int   -  period:      2^64
	case 1:
		return xoroshiro_next();
	default:
		xx = rotl(xx, 8) - rotl(xx, 29);                 //RERS,   period = 4758085248529 (prime)
		yy = rotl(yy, 21) - yy;  yy = rotl(yy, 20);      //RESR,   period = 3841428396121 (prime)
		zz = rotl(zz, 42) - zz;  zz = zz + rotl(zz, 14); //RESDRA, period = 5345004409 (prime)
		return xx ^ yy ^ zz;
	}
	//
	//
	/**/
}

/*
DIST_RangeN::DIST_RangeN(int seed, double a0, double a1) : 
	GRander(seed), rMin(a0), rMax(a1)  {	
	std::normal_distribution<> d1((rMax+rMin)/2,(rMax-rMin)/6);
	d=d1;
}

double DIST_RangeN::gen(){
	double a;
	do{
		a = d(g);
	} while (a<rMin || a>rMax);
	return (a);
}*/