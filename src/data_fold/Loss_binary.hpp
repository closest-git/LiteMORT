#pragma once

//#include "loss.hpp"

namespace Grusoft {
	class LambdaRank {
	protected:
		size_t nUnderFlow = 0, nOverFlow = 0;
		double a_0, a_1, grid;
		size_t nMost=-1;
		double sigma=20000;
		double *tables=nullptr;
	public:
		LambdaRank(int flag=0)	{}
		virtual ~LambdaRank() {
			if (tables != nullptr)
				delete[] tables;
		}
		virtual void Init(double sigma,double a_0,double a_1, size_t nMost, int flag = 0x0);
		virtual double At(double a);

	};
	




}

