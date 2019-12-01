#pragma once
#include <vector>
#include <math.h>
#include "./Histogram.hpp"
#include "../util/GST_def.h"
#include "../learn/DCRIMI_.hpp"
#include "../util/GRander.hpp"
#include "../util/PY_obj.hpp"

using namespace std;
#define ZERO_REGION 1.0e-16
#define IS_ZERO_RANGE(x)	( fabs(x)<ZERO_REGION )

namespace Grusoft {
	/*
		v0.1	cys
			和samp_set相关
	*/

	struct COR_RELATION {
		float *dcrimi = nullptr;	//注意，排序之后的
		double D_sum = 0;

		void Clear() {
			if (dcrimi != nullptr)			{
				delete dcrimi;	dcrimi = nullptr;
			}
		}
		virtual ~COR_RELATION() {
			Clear();
		}

		template<typename Tx, typename Ty>
		void DCRIMI_2(const LiteBOM_Config&config, Tx *val, Ty *y, const vector<tpSAMP_ID>& sort_ids, int flag = 0x0) {
			size_t i, jj,noBin = 0, pos, nA = sort_ids.size(), n_1, n_0, nA_0 = 0, nA_1 = 0;
			G_INT_64 i_0 = 0, i_1;
			D_sum = 0;
			double density_1, density_0,rWin=1.0/ config.feat_quanti/10,a0=val[sort_ids[0]], a1 = val[sort_ids[nA-1]];
			assert(a1>a0);
			double step = (a1 - a0)*rWin,a,cur;
			//int step = max(3, (int)(nA*rWin));
			for (i = 0; i < nA; i++) {
				y[i] == 1 ? nA_1++ : nA_0++;
			}
			if (nA_0 == 0 || nA_1 == 0)
				return;

			dcrimi = new float[nA]();
			for (i = 0; i < nA; i++) {
				pos = sort_ids[i];		cur = val[pos];
				i_0 = i_1 = i;			n_1 = 0, n_0 = 0;
				while (--i_0 >= 0) {
					pos = sort_ids[i_0];
					a = val[pos];
					if (a + step < cur)
						break;
					y[pos] == 1 ? n_1++ : n_0++;
				}
				while (++i_1<nA) {
					pos = sort_ids[i_1];
					a = val[pos];
					if (a - step > cur)
						break;
					y[pos] == 1 ? n_1++ : n_0++;
				}
				/*i_0 = i>step?i-step:0, i_1 = min(nA-1,i + step);
				for (n_1 = 0, n_0 = 0, jj = i_0; jj < i_1; jj++) {
					if (y[jj] == 1) {
						n_1++;
					}
					else {
						n_0++;
					}
				}*/
				density_1 = n_1*1.0 / nA_1, density_0 = n_0*1.0 / nA_0;
				dcrimi[i] = fabs(density_1 - density_0);
				D_sum += dcrimi[i];
			}
		}
	};

	/*
		吾道 一以贯之(Only init once before the train stage)
	*/
	struct Distribution {
		static GRander rander_;
		enum {
			CATEGORY = 0x100, DISCRETE = 0x200,
			V_ZERO_DEVIA = 0x10000,	//常值，一般可忽略
			DISTRI_OUTSIDE = 0x40000,
		};

		struct vDISTINCT{
			enum {
				BASIC,LARGE=0x10,TINY=0x20,
			};
			vDISTINCT(double v_,size_t n_) : val(v_),nz(n_)	{
			}
			double val=-1; 
			size_t nz=0;
			int type = BASIC;
		};			

		string nam,desc;
		vector<tpSAMP_ID> sortedA;		//排序后的有意义数据(NA-无意义数据)
		//vector<double>  vUnique;		
		vector<vDISTINCT>  vUnique;		//vThrsh
		MAP_CATEGORY mapCategory;
		HistoGRAM *histo = nullptr;

		int nHistoBin() {	return histo == nullptr ? 0 : histo->nBins;		}
		
		vector<BIN_FEATA> binFeatas;
		double split_F(int no, int flag=0x0) const;
		bool isUnique = false;
		bool isValidFeatas();

		size_t nSamp, nZERO = 0, nNA = 0;
		size_t type = 0x0;
		double vMin = DBL_MAX, vMax = -DBL_MAX,q1=-DBL_MAX,q2 = -DBL_MAX,q3 = -DBL_MAX;
		double H_q0 = 0, H_q1 = 0, H_q2 = 0, H_q3 = 0, H_q4 = 0;
		double rNA = 0, rSparse = 0;
		double mean = 0, median = 0, most_often = 0, devia = 0, impuri = 0;
		COR_RELATION corr;	//区分Y的能力估计

		virtual void ClearHisto() {
			//vThrsh.clear();
			vUnique.clear();
			if (histo != nullptr)			{
				delete histo;		histo = nullptr;
			}
		}
		virtual ~Distribution();

		virtual void Dump(int feat,bool isQuanti, int flag);

		Distribution() {}
		bool VerifySame(const Distribution& rhs) const;

		bool isPass()	const {
			if (rNA == 1)
				return true;
			if (ZERO_DEVIA(vMin, vMax))
				return true;

			return false;
		}

		void Merge(const Distribution& next) {
		}

		void STA_at(size_t N, const PY_COLUMN *col, bool isSparse, int flag) {
			STA_at(N, (float*)col->data, isSparse, flag);
		}

		/*
			不支持增量操作，多个数据应MERGE
		*/
		template<typename Tx>
		void STA_at(size_t N, const Tx *vec, bool isSparse, int flag) {
			nSamp = N;
			vMin = DBL_MAX, vMax = -DBL_MAX;
			mean = nan("");		median = nan("");
			nZERO = 0, nNA = 0;
			double a, a2 = 0, sum = 0, x_0, x_1;
			size_t i = 0, i_0 = 0, nA = 0;
			while (i_0 < N) {
				if (IS_NAN_INF(vec[i_0]))		{
					nNA++;	i_0++;
				}	else	{
					break;
				}
			}
			if (i_0 == N) {
				//printf("!!!All NA at dim=%d!!!\n", N);
				goto END;
			}
			x_0 = vec[i_0], x_1 = x_0;
			for (i = i_0; i < N; i++) {
				if (IS_NAN_INF(vec[i])) {
					nNA++;	continue;
				}
				if (IS_ZERO_RANGE(vec[i])) {
					nZERO++;
				}/**/
				a = vec[i];
				a2 += a*a;				sum += a;
				x_0 = MIN2(x_0, a);		x_1 = MAX2(x_1, a);
			}
			vMax = MAX2(vMax, x_1);				vMin = MIN2(vMin, x_0);
		END:
			if (1) {
				rNA = nNA*1.0 / N;					rSparse = nZERO*1.0 / N;
			}

			if (nNA > 0 && nNA < N && isSparse) {
				vector<Tx> A;
				vector<tpSAMP_ID> map;
				A.resize(N - nNA);
				map.resize(N - nNA);
				for (i = 0; i < N; i++) {
					if (IS_NAN_INF(vec[i])) {
						continue;
					}
					A[nA] = vec[i];	map[nA++] = i;
				}
				assert(N - nNA == nA);

				vector<tpSAMP_ID> idx;
				sort_indexes(A, idx);
				sortedA.resize(N - nNA);
				for (i = 0; i < nA; i++) {
					sortedA[i] = map[idx[i]];
				}
				for (i = 0; i < nA - 1; i++) {
					assert(!IS_NAN_INF(vec[sortedA[i]]));
					assert(vec[sortedA[i]] <= vec[sortedA[i + 1]]);
				}
			}

			if (N > nNA) {
				mean = sum / (N - nNA);
				impuri = a2 - (N - nNA)*mean*mean;
				if(impuri<0 && fabs(impuri)<1.0e-6*a2)
					impuri=0;
#ifdef _DEBUG
				assert(impuri >= 0);
#endif
				if (impuri < 0) {
					printf("!!!!!! impur=%g !!!!!!\n", impuri);
					devia = 0;		impuri = 0;
				}else
					devia = sqrt(impuri / (N - nNA));
			}
			else {
				assert(nNA == 0 || nNA == N);
			}
		}

		template<typename Tx>
		void STA_at(const vector<Tx>& vec, bool isSparse, int flag) {
			Tx *arr = (Tx*)(vec.data());
			size_t nSamp = vec.size(), i;
			STA_at(vec.size(), arr, isSparse, flag);
		}

		//允许重采样
		template<typename Tx>
		void EDA(const FeatsOnFold *hFold, size_t nSamp_, const SAMP_SET *samp_set, const Tx*samp_val_0,bool isGenHisto,int flag) {
			Tx *samp_val = (Tx*)samp_val_0;
			size_t i;
			if (samp_set != nullptr) {//EDA on replacement sampling
				nSamp_ = samp_set->nSamp;
				samp_val = new Tx[nSamp_];
				tpSAMP_ID *samps = samp_set->samps;
				for (i = 0; i < nSamp_; i++) {
					samp_val[i] = samp_val_0[samps[i]];
				}
			}
			/**/
			//hDistri->nam = nam;
			STA_at<Tx>(nSamp_, samp_val, true, 0x0);
			/*else if (config.eda_Normal != LiteBOM_Config::NORMAL_off) {
				Tx *val_c = arr();
				double mean = hDistri->mean, s = 1.0 / hDistri->devia;
				//for each(tpSAMP_ID samp in hDistri->sortedA
				for (i = 0; i < nSamp_; i++) {
				val_c[i] = (val_c[i] - mean)*s;
				}
				hDistri->STA_at(nSamp_, val, true, 0x0);
			}*/
			assert(histo == nullptr);
			if (isGenHisto) {	//Loss中的各个feat没必要生成histo
				X2Histo_(hFold->config, nSamp_, samp_val, (double*)nullptr);
			}

			//https://stackoverflow.com/questions/13944886/is-stdvector-memory-freed-upon-a-clear
			vector<tpSAMP_ID>().swap(sortedA);
			vector<Distribution::vDISTINCT>().swap(vUnique);
			if (samp_val != samp_val_0)
				delete[] samp_val;
		}

		template<typename Tx>
		void CheckUnique(LiteBOM_Config config, size_t nSamp_, const Tx *val, const vector<tpSAMP_ID>& idx, vector<vDISTINCT>& vUnique, /*int nMostUnique,*/ int flag = 0x0) {
			size_t nA = idx.size(), i;
			Tx a0 = val[idx[0]], a1 = val[idx[nA - 1]], pre = a0;
			size_t nz=1;
			//vUnique.push_back((double)a0);
			for (i = 1; i < nA; i++) {
				if (val[idx[i]] == pre)
				{		nz++;	 continue;		}
				assert(val[idx[i]] > pre);
				vUnique.push_back(vDISTINCT(pre, nz));		nz = 1;
				pre = val[idx[i]];
				/*if (vUnique.size() >= nMostUnique - 1) {
					vUnique.clear();	return;
				}*/
				//vUnique.push_back((double)pre);
			}
			vUnique.push_back(vDISTINCT(pre, nz));
			nz = 0;
			for (auto b : vUnique)	nz += b.nz;
			assert(nz == nA);
		}

		BIN_FEATA& AddBin(const LiteBOM_Config&config, size_t nz,double pre, double cur, int flag);		//返回FEATA，设置其它信息
		int HistoOnFrequncy_small(const LiteBOM_Config&config, vector<vDISTINCT>& vUnique, int i_0, int i_1, size_t T, int flag);
		//always last bin for NA
		void HistoOnFrequncy_1(const LiteBOM_Config&config, vector<vDISTINCT>& vUnique, size_t nA0, size_t nMostBin, int flag = 0x0);
		void HistoOnUnique_1(const LiteBOM_Config&config, vector<vDISTINCT>& vUnique, size_t nA0, bool isMap, int flag = 0x0);

		/*	
			v0.2	cys
				1/15/2019
			v0.3	cys
				4/8/2019
		*/
		template<typename Tx>
		void HistoOnFrequncy(const LiteBOM_Config&config, Tx *val, const vector<tpSAMP_ID>& sort_ids,
			size_t nMostBin, int flag = 0x0) {
			assert(histo != nullptr);
			size_t i, i_0 = 0, i_1, noBin = 0, pos, nA = sort_ids.size(), T_min_count = int(nA / nMostBin) + 1;
			Tx a0 = val[sort_ids[0]], a1 = val[sort_ids[nA - 1]], v0;
			double T_min_decrimi = 0,crimi=0;
			//vThrsh.clear();
			bool isDcrimi = corr.dcrimi != nullptr;
			if (isDcrimi) {
				T_min_count /= 2;  T_min_decrimi = corr.D_sum / nMostBin;
			}
			while (i_0 < nA) {
				v0 = val[sort_ids[i_0]];			i_1 = i_0;
				//noBin = vThrsh.size();
				if (isDcrimi) {
					crimi = corr.dcrimi[i_0];
				}
				HISTO_BIN& bin = histo->bins[noBin];
				bin.tic = noBin;	//tic split_F必须一致
				//bin.split_F = i_0 > 0 ? (v0 + val[sort_ids[i_0 - 1]]) / 2 : v0;
				/*//vThrsh.push_back(v1_last);
				if (i_0 > 0)
					vThrsh.push_back((v0 + val[sort_ids[i_0 - 1]]) / 2);
				else
					vThrsh.push_back(v0);
				*/
				while (++i_1 < nA) {
					pos = sort_ids[i_1];
					assert(!IS_NAN_INF(val[pos]));
					if( isDcrimi ){						
						if (i_1 - i_0 >= T_min_count && val[pos] > v0 && crimi > T_min_decrimi) {
							break;
						}
						crimi += corr.dcrimi[i_1];
					}else	if (i_1 - i_0 >= T_min_count && val[pos] > v0)
						break;
					v0 = val[pos];
				}
				assert(i_1 == nA || val[pos] > v0);
				bin.nz = i_1 - i_0;				
				i_0 = i_1;		
				noBin = noBin + 1;
			}
			//histo->bins.resize(noBin+1);
			histo->nBins = noBin + 1;
			assert(i_0 == nA);
			double delta = double(fabs(a1 - a0)) / nMostBin / 100.0;
			//histo->bins[noBin].split_F = a1 + delta;		//上界,为了等
			//assert(histo->bins[histo->bins.size()-1].split_F>a1);
			//vThrsh.push_back(a1 + delta);
		}

		/*
			v0.1
			v0.2	cys	
				10/10/2019
			The most common and popular approach is to model the missing value in a categorical column as a new category called “Unknown.”
		*/
		template<typename Tx>
		void HistoOnUnique(const LiteBOM_Config&config, Tx *val, const vector<tpSAMP_ID>& sort_ids, vector<vDISTINCT>&uniques, int flag = 0x0) {
			size_t nMostBin = uniques.size();
			assert(histo != nullptr);
			size_t i, i_0 = 0, i_1, noBin = -1, pos, nA = sort_ids.size(), T_min = int(nA / nMostBin) + 1;
			Tx a0 = val[sort_ids[0]], a1 = val[sort_ids[nA - 1]], v0;
			//vThrsh.clear();		
			binFeatas.resize(nMostBin);
			mapCategory.clear();		
			while (i_0 < nA) {
				v0 = val[sort_ids[i_0]];			i_1 = i_0;
				HISTO_BIN& bin = histo->bins[++noBin];
				BIN_FEATA& feata = binFeatas[noBin];
				bin.tic = noBin;
				mapCategory.insert(pair<int, int>((int)(v0), noBin));
				feata.split_F = noBin > 0 ? (v0 + uniques[noBin - 1].val) / 2 : v0;
				while (++i_1 < nA) {
					pos = sort_ids[i_1];
					assert(!IS_NAN_INF(val[pos]));
					if (val[pos] > v0)
						break;
					v0 = val[pos];
				}
				assert(i_1 == nA || val[pos] > v0);
				bin.nz = i_1 - i_0;
				i_0 = i_1;
			}
			assert(i_0 == nA);			assert(noBin== nMostBin-1);
			histo->nBins = noBin + 1;
			//AddBin(config, noBin + 1, a1, DBL_MAX, -0x0);	//last bin for NA
			binFeatas[noBin].split_F = DBL_MAX;
			size_t n1 = ceil(noBin / 4.0), n2 = ceil(noBin / 2.0), n3 = ceil(noBin *3.0 / 4);
			H_q0 = uniques[0].val,			H_q4 = uniques[noBin].val;
			H_q1 = q1 = uniques[n1].val,	H_q2 = q2 = uniques[n2].val;		H_q3 = q3 = uniques[n3].val;/**/

		}

		/*可以减少histo的bin数，但对准确率似乎没啥影响
			https://stats.stackexchange.com/questions/798/calculating-optimal-number-of-bins-in-a-histogram/862
		*/
		template<typename Tx>
		int Freedman_Diaconis_(const Tx * val, const vector<tpSAMP_ID>& idx,int flag=0x0) {
			size_t  nA = idx.size();
			assert(nA >= 4);
			Tx a0 = val[idx[0]], a1 = val[idx[nA - 1]], q3,q1;
			q3 = val[idx[nA * 3 / 4]], q1 = val[idx[nA / 4]];
			assert(q3>=q1);
			double IQR = q3-q1;
			if(IQR==0){
			//if (IQR <(a1-a0)/1000.0 ) {	//两端数据异常很常见啊
				IQR = a1 - a0;
			}
			double h = 2* IQR/pow(nA, 1.0 / 3);
			int nBin = (int)((a1 - a0) / h)+1;
			assert(nBin > 0);
			return nBin;
		}

		//只修改split_F
		virtual void UpdateHistoByW(const LiteBOM_Config&config, int nTree,float *wBins,int flag=0x0);

		/*
		必须保证histo与vThrsh严格一致		需要重新设计histo与vThrsh		3/11/2019
		Issue-Needed:
			how dividing the bins on the gradient statistics		http://mlexplained.com/2018/01/05/lightgbm-and-xgboost-explained/
		一些讨论
			https://github.com/Microsoft/LightGBM/issues/583
		*/
		template<typename Tx, typename Ty>
		void X2Histo_(const LiteBOM_Config&config, size_t nSamp_, Tx *val, Ty *y, int flag = 0x0) {
			if (rNA == 1.0) {
				printf("X2Histo_::!!!%s is NAN!!!\n", desc.c_str());
				return;
			}
			if (vMin == vMax) {
				printf("X2Histo_::%s is const(%g)!!!", desc.c_str(), val[0]);
				return;
			}			

			string optimal = config.leaf_optimal;
			assert(histo == nullptr);
			//histo = optimal == "grad_variance" ? new HistoGRAM(nSamp_) : new Histo_CTQ(nSamp_);
			histo = new HistoGRAM(nullptr,nSamp_);

			int nMostBin = config.feat_quanti;		
			//if(BIT_TEST(type,Distribution::DISCRETE))
			assert(nMostBin > 0);
			//nMostBin = max(1, config.feat_quanti / 4);
			vector<tpSAMP_ID> idx;
			if (sortedA.size() > 0)
				idx = sortedA;
			else
				sort_indexes(nSamp_, val, idx);
			size_t i, i_0 = 0, i_1, noBin = 0, pos, nA = idx.size();
			Tx a0 = val[idx[0]], a1 = val[idx[nA - 1]], v0 = a0;
			if (nA > 4 && a1>a0) {
				q1 = val[idx[nA / 4]], q2 = val[idx[nA / 2]], q3 = val[idx[nA*3/4]];
				//nMostBin = MIN2(nMostBin,Freedman_Diaconis_(val, idx));
			}	else {
				q1 = q2 = q3 = a0;
			}
			assert(a0 <= a1 && a0 == vMin && a1 == vMax);
			//histo->a0 = a0;		histo->a1 = a1;
			if (a0 == a1) { return; }/**/
			Tx step = (a1 - a0) / nMostBin, v1_last = a0;
			CheckUnique(config, nSamp_, val, idx, vUnique, nMostBin*10);
			if (BIT_TEST(type, Distribution::DISCRETE))
				nMostBin = vUnique.size() + 3;
			if (BIT_TEST(type, Distribution::CATEGORY) || BIT_TEST(type, Distribution::DISCRETE)) {
				if (vUnique.size() > 0) {
					assert(config.feat_quanti > 1);
					assert(histo->bins == nullptr);
					histo->bins = new HISTO_BIN[vUnique.size()+1];		binFeatas.resize(vUnique.size() + 1);
					HistoOnUnique_1(config, vUnique, nA, BIT_TEST(type, Distribution::CATEGORY));
					//if(config.isDebug_1)
					//else
					//	HistoOnUnique(config, val, idx, vUnique);
					//histo->Dump(this->binFeatas, mapCategory);		//输出Histogram的信息
					vUnique.clear();
					return;		//必须保持一致
				}
			}

			//int histo_alg = config.histo_algorithm;
			//histo->bins.resize(nMostBin + 3);
			assert(histo->bins==nullptr);
			histo->bins = new HISTO_BIN[nMostBin + 3];
			binFeatas.resize(nMostBin + 3);		
			switch (config.histo_bin_map) {
			case LiteBOM_Config::HISTO_BINS_MAP::onUNIQUE:
				
				break;
			case LiteBOM_Config::HISTO_BINS_MAP::on_FREQ_and_Y:
			case LiteBOM_Config::HISTO_BINS_MAP::on_FREQ:
				if( y!=nullptr && config.histo_bin_map== LiteBOM_Config::HISTO_BINS_MAP::on_FREQ_and_Y)
					corr.DCRIMI_2(config, val,y, idx,flag );
				/**/if (vUnique.size() <= nMostBin - 1) {	
					HistoOnUnique_1(config, vUnique, nA, false);
					//HistoOnUnique(config, val, idx, vUnique);
				}	else {
					HistoOnFrequncy_1(config, vUnique, nA, nMostBin-1);
				}
				corr.Clear();
				break;
				
			default:		//on_QUANTILE
				throw "!!!HISTO_BINS_MAP::on_QUANTILE is ...!!!";			
				break;
			}
			vUnique.clear();
			int nBin = histo->nBins;	// bins.size();		//always last bin for NA
			histo->nMostBins = histo->nBins;	// bins.size();
			assert(binFeatas.size()>=nBin );
			binFeatas.resize(nBin);
			//histo->bins.resize(nBin + 1);
			/*if (vUnique.size() > 0) {	//难道有BUG???
			}	else {

			}*/
			if (nBin >= 2) {
				size_t n1 = ceil(nBin / 4.0), n2 = ceil(nBin / 2.0), n3 = ceil(nBin *3.0 / 4) - 1;
				HISTO_BIN&b0 = histo->bins[0], &b1 = histo->bins[n1], &b2 = histo->bins[n2], &b3 = histo->bins[n3], &b4 = histo->bins[nBin - 1];
				//H_q0 = b0.split_F, H_q4 = b4.split_F;
				//H_q1 = q1 = b1.split_F, H_q2 = q2 = b2.split_F;		H_q3 = q3 = b3.split_F;/**/
			}
		}

	};
}