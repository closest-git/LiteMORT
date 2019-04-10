
#include "GRander.hpp"

using namespace Grusoft;

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