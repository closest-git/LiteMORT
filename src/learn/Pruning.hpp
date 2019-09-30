#pragma once

#include <string>
#include <vector>
#include <time.h>
#include <omp.h>
#include "../util/samp_set.hpp"


namespace Grusoft{
	typedef double tpMetricU;
	class ManifoldTree;
	class FeatsOnFold;
	class EnsemblePruning{
		double *orth=nullptr;
		double *gamma = nullptr;
		int num_orth=0,ldOrth=0;
	protected:
		static bool isDebug,isRand;
		FeatsOnFold *hFold = nullptr;
		BoostingForest *hBoost = nullptr;

		//|wx|=1	|wy|=1 
		tpMetricU *mA = nullptr, *ax_=nullptr;
		tpMetricU *mB = nullptr, *wy=nullptr;		//live section of (A,x)
		int ldA_;	//row_major
		int nLive = 0, nLive_0 = 0;
		int *wasSmall=nullptr, *isLive = nullptr, *wasLive=nullptr,*y2x=nullptr;
		std::vector<tpSAMP_ID> sorted_indices;
		int nSparsified() {
			int nPick, i;
			for (nPick = 0, i = 0; i < nWeak; i++) {
				if (wx[i] > 0)
					nPick++;
			}
			return nPick;
		}
		//is_live=abs_x < 1.0-delta;	 mB = mA[:,is_live];		wy = wx[is_live]
		int SubOnLive(double delta,bool update_orth,double *v_0,double *v_sub,int flag);

		void ToCSV(const string& sPath, int flag);
		void LoadCSV(const string& sPath, int flag);
		double UpateGamma(int *isLive, int nY,int flag = 0x0);
		bool partial_infty_color(int nX,bool balance, int flag = 0x0);
		void sorted_ax(int flag=0x0);
		void make_orthogonal(tpMetricU *b, int ldB, int &nRun, int nMost, int nLive_0, int *isSmall, int flag=0x0);
		void basic_local_search(double *,bool balanced = false, int flag = 0x0);
		void local_improvements(double *, bool balanced = false, int flag = 0x0);
		void greedy(double*,bool balanced = false, int flag = 0x0);
		void round_coloring(bool balanced = false, int flag=0x0);
		virtual void Prepare(int flag = 0x0);
	public:
		size_t nSamp = 0, nWeak = 0, nMostWeak = 0;
		int nPruneOperation = 0;
		tpMetricU *init_score = nullptr;
		std::vector<ManifoldTree*>forest;
		double *plus_minus = nullptr;
		//combination coefficient
		tpMetricU *cc_0 = nullptr, *cc_1 = nullptr,cc_0_sum=0, *wx = nullptr;

		EnsemblePruning(BoostingForest *hBoost,FeatsOnFold *hFold, int nWeak_,int flag=0x0);
		virtual ~EnsemblePruning();
		virtual bool isValid() { return true; }

		virtual void Reset4Pick(int flag);
		virtual bool Pick(int nWeak_,int isToCSV, int flag);
		virtual bool Compare( int flag);

		virtual void OnStep(ManifoldTree *hTree, tpDOWN*down, int flag = 0x0);
	};

};


