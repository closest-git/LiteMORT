#pragma once

#include <memory>
#include <string>
#include <vector>
#include <algorithm>
#include <numeric> 
#include "../util/GST_def.h"
#include "../include/LiteBOM_config.h"
#include "./Histogram.hpp"
#include "./Distribution.hpp"
#include "Imputer.hpp"

using namespace std;


//#include "DataFold.hpp"
namespace Grusoft {
	template<typename Tx>
	int NA2mask(size_t nMostDup, Tx *feat, size_t nSamp_, int stp, int*mask, size_t &nnz, int flag = 0x0) {
		size_t i, nDup = 0;
		for (nDup = 0, i = 0; i < nSamp_; i++) {
			//if (feat[i] == 0)	continue;
			if (IS_NAN_INF(feat[i]))
				continue;
			if (mask[i] == stp) {
				nDup++;
				if (nDup>nMostDup)
					break;
			}
			else {
				mask[i] = stp;		nnz++;
			}
		}
		return nDup;
	}

	class FeatsOnFold;
	class ExploreDA{
	protected:
		//int nFeat=0;
	public:
		typedef map<int, Distribution*> MAP_DISTRI;
		static bool isZERO(double len) {
			return len==0;
		}
		struct Bundle {
			//int nz=0;
			size_t nMostDup=0;
			vector<vector<int>> buns;
			vector<double> off;
		};
		Bundle bundle;
		MAP_DISTRI mapDistri;
		int nFeat() {	return mapDistri.size();	}
		void AddDistri(const PY_COLUMN*col, int id, int flag = 0x0);
		Distribution* GetDistri(int id)	;
		//vector<Distribution> arrDistri;

		ExploreDA( LiteBOM_Config&,int flag=0x0);
		
		virtual ~ExploreDA();

		virtual void CheckDuplicate(LiteBOM_Config config, int flag) {
		}

		template<typename Tx, typename Ty>
		void Analysis(LiteBOM_Config config, Tx *X_, Ty *Y_, size_t nSamp_, size_t ldX_, size_t ldY_, int flag) {
			clock_t t0=clock();
			printf("********* EDA::Analysis nSamp=%ld nFeat=%d........\n", nSamp_, nFeat);
			assert(ldX_==nFeat);
			size_t feat,i, nConstFeat=0;
			double sparse=0, nana=0;
			vector<double> arrD;
			for (feat = 0; feat < ldX_; feat++) {
				if(feat==2)
				{	feat = 2;	}
				Distribution &distri = arrDistri[feat];
				distri.desc = "distri_" + std::to_string(feat);
				Tx *x=X_ + feat*nSamp_,a2,sum,x_0,x_1;
				distri.STA_at(nSamp_,x,true,0x0);
				sparse+= distri.rSparse*nSamp_;
				nana += distri.rNA*nSamp_;
				if (distri.rNA == 1.0) {
					printf("---EDA---\t!!!Feat_%ld is NAN!!!\n", feat);
				}
				if(ZERO_DEVIA(distri.vMin, distri.vMax) )
				{	nConstFeat++;	}				
				distri.X2Histo_(config,nSamp_, x, Y_);
				distri.Dump(feat,false,flag);		//输出distribution信息
				arrD.push_back(distri.corr.D_sum);
			}
			if (nFeat > 9) {	//
				vector<int> idx,feats;
				idx.resize(ldX_);// initialize original index locations
				std::iota(idx.begin(), idx.end(), 0);
				// sort indexes based on comparing values in v
				std::sort(idx.begin(), idx.end(), [&arrD](int i1, int i2) {return arrD[i1] > arrD[i2]; });
				feats.push_back(idx[0]);	feats.push_back(idx[1]);	feats.push_back(idx[2]);	
				feats.push_back(idx[3]);	feats.push_back(idx[4]);	feats.push_back(idx[5]);
				feats.push_back(idx[nFeat-3]);	feats.push_back(idx[nFeat-2]);	feats.push_back(idx[nFeat-1]);
				printf("DCRIMI:\t");
				for (auto feat : feats) {
					printf("%.4g(%d) ", arrD[feat], feat);
				}
			}
			sparse/=(nSamp_*ldX_);		nana /= (nSamp_*ldX_);
			printf("\n********* EDA::Analysis const=%ld sparse=%g NAN=%g T=%.3g........OK\n",  nConstFeat, sparse, nana, (clock() - t0) / 1000.0);
		}
		

		/*
			原则上只适用于稀疏的数值特征！！！
		*/
		template<typename Tx>
		void InitBundle(LiteBOM_Config config, Tx *X_, size_t nSamp_, size_t ldX_, int flag) {			//GST_TIC(tick);

			assert(ldX_ == nFeat);
			size_t feat, i,seed;
			int *mask=new int[nSamp_](),stp=0,nDup=0, *used=new int[nFeat](),nMerge=0, nPass=0;
			bundle.nMostDup = nSamp_/10000;
			printf("********* GBRT::InitBundle nSamp=%lld nFeat=%d nMostDup=%d......\n", nSamp_, nFeat, bundle.nMostDup);
			for (seed = 0; seed < nFeat; seed++) {
				if (arrDistri[seed].isPass()) {
					used[seed] = 1;		nPass++;
				}
			}
			for (seed = 0; seed < nFeat; seed++) {			
				if(used[seed] == 1)
					continue;
				printf("\r\tBundle@%d ........", seed);

				vector<int> bun;
				size_t nnz=0;
				stp++;
				nDup = NA2mask(bundle.nMostDup,X_ + seed*nSamp_,  nSamp_, stp, mask, nnz);		assert(nDup==0);
				for (feat = 0; feat < nFeat; feat++) {	
					if(feat==seed || used[feat]==1)
						continue;
					nDup = NA2mask(bundle.nMostDup,X_ + feat*nSamp_, nSamp_, stp, mask, nnz);
					if (nDup <= bundle.nMostDup) {
						bun.push_back(feat);
						used[feat]=1;
					}
				}
				if (bun.size() > 0) {
					bun.push_back(seed);		nMerge+=bun.size();
					std::sort(bun.begin(),bun.end());
					assert(nnz<=nSamp_);
					bundle.buns.push_back(bun);
					printf("\tBundle@%d nFeat=[%d] nnz=%lld........\n", seed,bun.size(),nnz);

				}
			}
			printf("\n********* bundles=%d nMerge=%d nPass=%d\n", bundle.buns.size(), nMerge, nPass);
			delete[] mask;		delete[] used;
				
		}

		
	};
	


}