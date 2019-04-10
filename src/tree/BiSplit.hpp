#pragma once

#include <vector>
#include <random>
#include <algorithm>
#include <assert.h>
#include <time.h>

#include "../util/GST_def.h"
#include "../util/samp_set.hpp"
#include "../data_fold/Histogram.hpp"
#include "../data_fold/Binfold.hpp"
#include "../learn/Regression.hpp"

using namespace std;



/*
	WeakLearner理解有误，实为BiSplit
*/
namespace Grusoft {

	class FeatVector;
	class FeatsOnFold;
	class BoostingForest;		//BoostingForest需要改名为BOModel

	/*
		训练时写入，train,eval,predict都会读取该值，而且该值有不同的表示
		tpFRUIT_应该为模板类
		SetSplitInfo需要重新设计！！！
	*/


	class MT_BiSplit {
	protected:
		//FeatsOnFold *hData=nullptr;
		virtual double AGG_CheckGain(FeatsOnFold *hData_, FeatVector *hFeat, int flag = 0x0);
		virtual int PickOnGain(FeatsOnFold *hData_, const vector<FRUIT *>& arrFruit, int flag = 0x0);

	public:
		static double tX;
		tpDOWN down_step;	//ManifoldTree的目标函数是negative_gradient
		double lr_eta=1.0;		//adaptive learning rate
		//HistoGRAM histo;
		Regression *regression = nullptr;
		BinFold *bsfold=nullptr;
		FRUIT *fruit = nullptr;		//only ONE!!!		thrsh at fork, Y value at leaf
		std::string sX;

		int id;				//为了节省内存，很难想象fold会超过65536
		float *distri = nullptr;
		SAMP_SET samp_set;
		MT_BiSplit	*left = nullptr, *right = nullptr;
		//size_t nzSamp = 0;
		int feat_id = -1, feat_regress = -1, depth = -1;
		double gain = 0, confi = 0, devia = 0;
		union {
			double impuri = DBL_MAX;
			double score ;
		};
		double Y_sum = 0, Y2_sum = 0, G_sum = 0, H_sum = 0;

		MT_BiSplit(FeatsOnFold *hData_, int d, int rnd_seed, int flag = 0x0);
		MT_BiSplit() : feat_id(-1), gain(0) { ; }
		MT_BiSplit(MT_BiSplit *hDad, int flag = 0x0) {
			depth = hDad->depth + 1;
			assert(hDad->feat_id != -1);
			feat_regress = hDad->feat_id;
		}
		virtual ~MT_BiSplit() {
			if (regression != nullptr)
				delete regression;
			if (fruit != nullptr)
				delete fruit;
			if (distri != nullptr)
				delete distri;
		}

		virtual size_t nSample() { return samp_set.nSamp; }
		bool isLeaf() const {
			return left == nullptr && right == nullptr;
		}

		//样本只是某个泛函的观测值!!!
		virtual void Observation_AtLocalSamp(FeatsOnFold *hData_, int flag = 0x0);
		virtual double CheckGain(FeatsOnFold *hData_, const vector<int> &pick_feats, int x, int flag = 0x0);
		virtual double GetGain(int flag = 0x0);
		virtual void BeforeTrain(FeatsOnFold*, int flag = 0x0) { throw "MT_BiSplit::BeforeTrain is ..."; }
		//virtual void SetSamp_(FeatsOnFold *hData_);
		virtual void Dump(const char*, int type, int flag = 0x0);
		static bool isBig(const MT_BiSplit *l, const MT_BiSplit *r) { return true; }/**/


		virtual tpDOWN GetDownStep() {
			assert(this->isLeaf());
			return down_step*lr_eta;
		}

		virtual void Init_BFold(FeatsOnFold *hData_,int flag=0x0);

		template<typename Tx>
		void SplitOn(FeatsOnFold *hData_, const std::vector<Tx>&vals, bool isQuanti, int flag = 0x0) {
			SAMP_SET& lSet = left->samp_set;
			SAMP_SET& rSet = right->samp_set;
			lSet = this->samp_set;		rSet = this->samp_set;
			lSet.isRef = true;			rSet.isRef = true;
			size_t i, nSamp = samp_set.nSamp;
			size_t &nLeft = samp_set.nLeft;
			size_t &nRigt = samp_set.nRigt;
			tpSAMP_ID samp, *samps = samp_set.samps,*rigt= samp_set.rigt,*left= samp_set.left;
			assert(nLeft == 0 && nRigt == 0);
			nLeft = 0;		nRigt = 0;
			//double thrsh = isQuanti ? fruit->T_quanti : fruit->thrshold;
			double thrsh = fruit->Thrshold(isQuanti);
			//clock_t t1 = clock();
			if (false) {	//std::sort确实要慢啊
				std::sort(samps, samps+ nSamp, [&vals](size_t i1, size_t i2) {return vals[i1] < vals[i2]; });
				for (i = 0; i < nSamp; i++) {
					samp = samps[i];
					//if (i > 0)	assert(vals[samp]>= vals[samps[i-1]]);
					if (vals[samp] < thrsh)
						nLeft++;
					else		{		break;		}
				}
				nRigt = nSamp - nLeft;
			}			else {
				for (i = 0; i < nSamp; i++) {
					samp = samps[i];
					//教训啊，排序能有效提升速度		3/27/2019
					//if (i > 1)		assert(samp>samps[i-1]);
					bool isNana = (isQuanti && vals[samp] == -1) || (!isQuanti && IS_NAN_INF(vals[samp]));
					if (isNana) {
						if (fruit->isNanaLeft) {
							samps[nLeft++] = samp;	continue;
						}	else {
							rigt[nRigt++] = samp;	continue;
						}
					}	else {
						if (vals[samp] < thrsh)
							samps[nLeft++] = samp;
							//left[nLeft++] = samp;
						else
							rigt[nRigt++] = samp;
					}
				}
				//memcpy(samps, samp_set.left, sizeof(tpSAMP_ID)*nLeft);
				memcpy(samps + nLeft, samp_set.rigt, sizeof(tpSAMP_ID)*nRigt);
			}
			//tX += ((clock() - (t1))*1.0f / CLOCKS_PER_SEC);
			lSet.samps = samps;				lSet.nSamp = nLeft;
			//std::sort(lSet.samps, lSet.samps + lSet.nSamp);
			rSet.samps = samps + nLeft;		rSet.nSamp = nRigt;
			//std::sort(rSet.samps, rSet.samps + rSet.nSamp);
		}
	};
	typedef MT_BiSplit *hMTNode;
	typedef std::vector<MT_BiSplit*>MT_Nodes;

}

