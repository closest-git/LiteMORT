#pragma once

#include <string>
#include <vector>
#include <time.h>
#include <omp.h>
#include "../util/samp_set.hpp"


namespace Grusoft{
	typedef double tpMetricU;
	class FeatsOnFold;
	class EnsemblePruning{
		double *orth=nullptr;
		double *gamma = nullptr;
		int num_orth=0,ldOrth=0;
	protected:
		//|wx|=1	|wy|=1 
		tpMetricU *mA = nullptr, *ax_=nullptr;
		tpMetricU *mB = nullptr, *wy=nullptr;		//live section of (A,x)
		int ldA;	//ͳһΪrow_major
		int *wasSmall=nullptr, *isLive = nullptr, *wasLive=nullptr,*y2x=nullptr;
		std::vector<tpSAMP_ID> sorted_indices;
		size_t nSamp=0, nWeak=0,nMostWeak=0;
		FeatsOnFold *hFold = nullptr;
		int nSparsified() {
			int nPick, i;
			for (nPick = 0, i = 0; i < nWeak; i++) {
				if (wx[i] > 0)
					nPick++;
			}
			return nPick;
		}

		int SubOnLive(int flag);
		void ToCSV(const string& sPath, int flag);
		void LoadCSV(const string& sPath, int flag);
		double UpateGamma(int *isLive, int nY,int flag = 0x0);
		bool partial_infty_color(int nX,bool balance, int flag = 0x0);
		void sorted_ax(int flag=0x0);
		void make_orthogonal(tpMetricU *b, int ldB, int &nRun, int nMost, int nLive_0, int *isSmall, int flag=0x0);
	public:
		short *plus_minus = nullptr;
		tpMetricU *w_0 = nullptr, *wx = nullptr;

		EnsemblePruning(FeatsOnFold *hFold, int nWeak_,int flag=0x0);
		virtual ~EnsemblePruning();
		virtual bool isValid() { return true; }

		virtual void Pick(int nWeak_,int T, int flag);

		virtual void OnStep(int noT, tpDOWN*down, int flag = 0x0);
	};

};


