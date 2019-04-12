#pragma once
#include <vector>
#include <float.h>
#include <string.h>
#include <assert.h>
#include "Parallel_t.hpp"

using namespace std;

typedef int tpSAMP_ID;			//SAMP_SET需要节省内存
//typedef std::vector<tpSAMP_ID> SAMP_SET;

namespace Grusoft {
class FeatsOnFold;
class SAMP_SET {
	void clear() {
		if (!isRef) {
			if (root_set != nullptr)	delete[] root_set;
			if (left != nullptr)		delete[] left;
			if (rigt != nullptr)		delete[] rigt;
		}
		nSamp = 0, nLeft = 0, nRigt = 0;
		Y_sum_1 = 0;
		Y_sum_2 = 0, Y_0 = DBL_MAX, Y_1 = -DBL_MAX;

		root_set = nullptr;
	}
public:
	bool isRef = true;
	tpSAMP_ID *root_set = nullptr;
	//samps当前节点，总是指向root_set的某个位置
	tpSAMP_ID *samps=nullptr,*left = nullptr, *rigt = nullptr;
	size_t nSamp = 0, nLeft=0,nRigt=0;
	//实际上是down vector的sum_1, 参见samp_set.STA_at<tpDOWN>(down, a2, mean, y_0, y_1,true)
	double Y_sum_1 = 0;
	double Y_sum_2 = 0, Y_0 = DBL_MAX, Y_1 = -DBL_MAX;
	/*tpSAMP_ID *samps_() {
		return set + off;
	}*/
	SAMP_SET( )	{}

	//很重要，原则上每棵树的样本可以任意重设
	virtual void SampleFrom(FeatsOnFold *hData_, const SAMP_SET *,size_t nMost,int rnd_seed, int flag = 0x0);
	void Alloc(size_t nSamp_, int flag = 0x0) {
		clear(); 

		isRef = false;
		nSamp = nSamp_;
		root_set = new tpSAMP_ID[nSamp];
		left = new tpSAMP_ID[nSamp];				rigt = new tpSAMP_ID[nSamp];
		for (size_t i = 0; i<nSamp; i++)		
			root_set[i] = i;
		samps = root_set;
	}
	void ClearInfo() {
		nSamp = 0;		nLeft = 0, nRigt = 0;
		samps = nullptr, left = nullptr, rigt = nullptr;
	}/**/


	virtual ~SAMP_SET() {
		clear();
		/*if (!isRef) {
			if (root_set != nullptr)	delete[] root_set;
			if (left != nullptr)		delete[] left;
			if (rigt != nullptr)		delete[] rigt;
		}*/
	}
	
	template<typename Tx, typename Ty>
	void SplitOn(const std::vector<Tx>&vals, const Ty& thrsh, SAMP_SET& lSet, SAMP_SET& rSet,int flag = 0x0) {
		lSet = *this;		rSet = *this;
		size_t i;
		tpSAMP_ID samp;
		assert(nLeft==0 && nRigt==0);
		nLeft = 0;		nRigt=0;
		if (flag == 1) {
			for (i = 0; i<nSamp; i++) {
				samp = samps[i];
				if (vals[i] < thrsh)
					left[nLeft++] = samp;
				else
					rigt[nRigt++] = samp;
			}
		}	else/**/ {
			for (i=0;i<nSamp;i++) {
				samp = samps[i];
				//assert (folds[samp] == fold_seed);
				if (vals[samp] < thrsh)
					left[nLeft++] = samp;
				else
					rigt[nRigt++] = samp;
			}
		}
		
		memcpy(samps,left,sizeof(tpSAMP_ID)*nLeft );
		memcpy(samps + nLeft, rigt, sizeof(tpSAMP_ID)*nRigt);
		lSet.samps=samps;				lSet.nSamp= nLeft;
		rSet.samps = samps+nLeft;		rSet.nSamp = nRigt;		
	}

	template<typename Tx, typename Ts>
	void Update(std::vector<Tx>&val, const Ts& step, int flag = 0x0) {
		tpSAMP_ID samp;
		size_t i;
		for (i = 0; i<nSamp; i++) {
			samp = samps[i];
			val[samp] += step;		//对应于UpdateResi
		}
	}

	//v0.2	parallel
	template<typename Tx>
	void STA_at(const Tx *vec, Tx&a2_, Tx&sum_, Tx&x_0, Tx&x_1,bool hasY) {
		size_t step;
		Tx a2 = 0,sum = 0;
		x_0 = vec[samps[0]], x_1 = x_0;
		int num_threads = OMP_FOR_STATIC_1(nSamp, step,1024);
#pragma omp parallel for schedule(static,1) reduction(+ : a2,sum)
		for (int thread = 0; thread < num_threads; thread++) {
			size_t start = thread*step, end = min(start + step, nSamp), i;
			Tx local_0 = vec[samps[start]], local_1 = local_0,a;
			tpSAMP_ID samp;
			for (i = start; i < end; i++) {
			//for (i = 0; i<nSamp; i++) {
				samp=samps[i];
				a = vec[samp];
				//if(IS_NAN_INF(a))
				//{	continue;	}
				a2 += a*a;				sum += a;
				local_0 = MIN(local_0, a);		local_1 = MAX(local_1, a);
			}
#pragma omp critical
			{	x_0 = MIN(local_0, x_0);			x_1 = MAX(local_1, x_1);	}
		}
		a2_ = a2;	sum_ = sum;
		if (hasY) {
			Y_sum_1 = sum;		Y_sum_2 = a2;
			Y_0 =x_0,			Y_1 =x_1;
		}
	}

	
	
};
};
