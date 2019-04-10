#pragma once

#include "DataFold.hpp"
#include "../util/samp_set.hpp"

namespace Grusoft {	
	/*
		v0.1	cys
			3/3/2019
	*/
	//template <typename Tx>
	class Move_Accelerator {
	protected:
		size_t nzParam = 0,nzMost;
		tpDOWN *velocity = nullptr,*alpha_p=nullptr,*deltX=nullptr,*gPast=nullptr;
		double friction = 0.5;
	public:
		typedef enum {
			MOVE_SGD = 0, MOVE_BFGS = 110, MOVE_HFNK = 111, MOVE_SGD_m, MOVE_SGD_n, SGD_A_g, SGD_A_dx,
		}ALGORITHM;
		ALGORITHM alg = MOVE_SGD_n;

		Move_Accelerator(int flag=0x0) {

		}
		virtual ~Move_Accelerator() {
			if (velocity != nullptr)			delete[] velocity;
			if (alpha_p != nullptr)				delete[] alpha_p;
		}

		template<typename Tx, typename Ty>
		void Init_T(size_t nMost_, int flag=0) {
			nzMost = nMost_;
			velocity = new tpDOWN[nMost_]();
			alpha_p = new tpDOWN[nMost_];
		}


		virtual void BeforeStep(const SAMP_SET&samp_set, tpDOWN * allx, int flag=0x0) {
			size_t nzParam = samp_set.nSamp;
			const tpSAMP_ID *samps = samp_set.samps;
			tpSAMP_ID no;
			assert(nzParam<= nzMost);
			size_t i=0;
			for (i = 0; i < nzParam; i++) {
				no = samps[i];
				alpha_p[no] = allx[i];
			}
		}
		virtual void AfterStep(const SAMP_SET&samp_set,tpDOWN * allx, int flag = 0x0) {
			size_t nzParam = samp_set.nSamp;
			const tpSAMP_ID *samps = samp_set.samps;
			assert(nzParam <= nzMost);
			size_t i,pos;
			for (pos = 0; pos < nzParam; pos++) {
				i = samps[pos];
				alpha_p[i] = allx[i] - alpha_p[i];
				allx[i] -= alpha_p[i];
			}
			Update(samp_set,allx,alpha_p, flag);
		}

		/*
			参见"Historical Gradient Boosting Machine"			 Experimental results show that our approach improves the convergence speed
				"Accelerated proximal boosting"
						of gradient boosting without significant decrease in accuracy.
		*/
		double Update(const SAMP_SET&samp_set, tpDOWN * allx, tpDOWN *alpha_p,  int flag) {
			size_t nzParam = samp_set.nSamp;
			const tpSAMP_ID *samps = samp_set.samps;
			assert(nzParam <= nzMost);
			//	printf( "%g\t",alph );
			size_t i, pos;
			double rou = 0.99, a, dx, one = 1.0;			
			switch (alg) {
			case MOVE_SGD_m:
				//		friction = 0.5;	
				for (pos = 0; pos < nzParam; pos++) { 
					i = samps[pos];
					velocity[i] = friction*velocity[i] + alpha_p[i]; 
				}
				for (pos = 0; pos < nzParam; pos++) {
					i = samps[pos];
					allx[i] += velocity[i];
				}
				break;
			case MOVE_SGD_n:
				if (1) {
					//AXPBY(nzParam, (FLOA)alph, p, (FLOA)friction, velocity);		
					//AXPY(nzParam, (FLOA)alph, p, allx);							
					//AXPY(nzParam, (FLOA)friction, velocity, allx);	
					for (pos = 0; pos < nzParam; pos++) {
						i = samps[pos];
						velocity[i] = friction*velocity[i] + alpha_p[i];
						allx[i] += alpha_p[i];
						allx[i] += friction*velocity[i];
					}
				}
				else {	//cs231n.github.io里的另一个公式 
					/*AXPY(nzParam, (FLOA)(-friction), velocity, allx);							
					//allx[i] += alph*p[i];
					AXPBY(nzParam, (FLOA)alph, p, (FLOA)friction, velocity);		
					//velocity[i] = friction*velocity[i]+alph*p[i];
					AXPY(nzParam, (FLOA)(1 + friction), velocity, allx);							
					//allx[i] += alph*p[i];*/
				}

				break;
			case SGD_A_dx:
				/*GST_THROW( "gPast is nullptr@5/8/2016" );
				for (i = 0; i < nzParam; i++) {
					gPast[i] = rou*gPast[i] + (1 - rou)*p[i] * p[i];
					a = -sqrt(deltX[i] + 1.0e-6) / sqrt(gPast[i] + 1.0e-6);
					dx = allx[i];		 allx[i] += a*p[i];		dx = allx[i] - dx;
					deltX[i] = rou*deltX[i] + (1 - rou)*dx*dx;
				}*/
				break;
			case SGD_A_g:		//adaptive subgradient
				/*				
				for (i = 0; i < nzParam; i++) {
					gPast[i] += p[i] * p[i];
					allx[i] += alph*p[i] / sqrt(gPast[i] + 1.0e-6);
				}*/
				break;
			default:
				for (pos = 0; pos < nzParam; pos++) {
					i = samps[pos];
					allx[i] += alpha_p[i];
				}
				break;
			}
			
			//LOSS()->OnUpdate(alph, flag);		

			return 0.0;
		}
	};




}

