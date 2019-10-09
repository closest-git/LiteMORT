#pragma once

#include <string>
#include <vector>
#include <time.h>
#include <omp.h>
#include <numeric> 
#include "../util/samp_set.hpp"

struct DataSample;

namespace Grusoft{
	struct DCRIMI_ {
		//static double tX;
		typedef enum {
			G_DCRIMI_1
		}TYPE;
		bool isSaveFalse = false;
		bool isBidui = false;
		//arrPFNO fnoFalse;
		TYPE type = G_DCRIMI_1;
		int D_span = 1000;
		float *D_inter = nullptr, *D_intra = nullptr;

		void *hBase = nullptr;
		double rFAR = 0, rFRR = 0, rEER = 0, D_, sep = 0, eer_sep = 0;
		double f_ar_8[8], f_rr_8[8], hd_8[8];	//对应1.0e-7~1.0的8个刻度
		double CRR = 0, time = 0;
		double nz_a = 0, nz_r = 0, mean_a = 0, mean_r = 0, devia_a = 0, devia_r = 0, max_a = 0, min_a = 0, max_r = 0, min_r = 0;
		double rTop_1 = 0, rTop_5 = 0;

		int epoch, dump = 0;
		std::string sTime;

		DCRIMI_(const DCRIMI_& dcri);
		DCRIMI_(void *hB, int D_span = 1000, int flag = 0x0);
		virtual ~DCRIMI_() {
			if (D_inter != nullptr)		delete[] D_inter;
			if (D_intra != nullptr)		delete[] D_intra;
		}
		void Insert_1(float dis, bool isIntra, int flag = 0x0);
		void Init(int flag = 0x0);
		void Analyze(const std::string &sTitle, int flag = 0x0);
		void GetRoc(float *roc, int flag = 0x0);
		double T_intra(int flag = 0x0);
	};
	typedef std::vector<DCRIMI_*> arrDCRIMI;

	struct DCRIMI_2 {
		static double tX;
		size_t N_pos = 0, N_neg = 0,dim=0;
		double N_0, N_1, P_0 , P_1;
		//tpSAMP_ID *samp_1 = nullptr;
		DCRIMI_2() { ; }
		virtual ~DCRIMI_2() {
			//if (samp_1 != nullptr)		
			//	delete[] samp_1;
		}

		template<typename Tx>
		void StatAtLabel(size_t dim_, const Tx *label, const Tx *y1, int flag = 0x0) {
			dim = dim_;
			N_pos = 0;		N_neg = 0;
			size_t i;
			Tx n_0 = y1[0], n_1 = y1[0], p_0 = y1[0], p_1 = y1[0], a;
			for (i = 0; i < dim; i++) {
				for (i = 0; i < dim; i++) {
					a = y1[i];
					if (label[i] == 1) {
						N_pos++;
						p_0 = min(p_0, a);		p_1 = max(p_1, a);
					}
					else {
						n_0 = min(n_0, a);		n_1 = max(n_1, a);
					}
				}
			}
			N_0 = n_0,		N_1 = n_1,		P_0 = p_0, P_1 = p_1;
			N_neg = dim - N_pos;
		}
		template<typename Tx>
		double AUC_Jonson(size_t dim, const Tx *y0, const Tx *y1, int flag = 0x0) {
			clock_t t1 = clock();
			/*	测试用例，auc=0.27472527472527464
			dim = 20;
			Tx y1[] = { 7.96542987e-01, 1.83434790e-01, 7.79691000e-01, 5.96850158e-01, 4.45832753e-01, 9.99749158e-02, 4.59248892e-01, 3.33708611e-01, 1.42866818e-01, 6.50888473e-01, 5.64115790e-02, 7.21998772e-01, 9.38552709e-01, 7.78765841e-04, 9.92211559e-01, 6.17481510e-01, 6.11653160e-01, 7.06630522e-03, 2.30624250e-02, 5.24774660e-01 };
			Tx y0[] = { 0., 0., 0., 0., 1., 1., 0., 1., 1., 0., 0., 0., 0., 0., 1., 0., 0., 1., 1., 0. };
			
			size_t i, N_pos=0, N_neg=0;
			for (i = 0; i < dim; i++) {
				if (y0[i] == 1)	N_pos++;
			}
			N_neg = dim - N_pos;*/
			double dot=0, auc =0;
			StatAtLabel(dim, y0, y1, flag);
			if (N_pos == 0) {
				auc = 1;
			}else if (N_neg == 0) {
				auc = 1;
			}	else {
				vector<tpSAMP_ID> idx;
				size_t i;
				idx.resize(dim);// initialize original index locations
				std::iota(idx.begin(), idx.end(), 0);
				std::sort(idx.begin(), idx.end(), [&y1](tpSAMP_ID i1, tpSAMP_ID i2) {return y1[i1] > y1[i2]; });
				//ParallelSort(idx.begin(), idx.end(), [&y1](tpSAMP_ID i1, tpSAMP_ID i2) {return y1[i1] > y1[i2]; }, IteratorValType(idx.begin()));
				for (i = 0; i < dim; i++) {		//首先对score从大到小排序，然后令最大score对应的sample 的rank为n
					dot += y0[idx[i]]*(i+1);
				}
				auc = 1 + ((N_pos + 1.) / (2 * N_neg)) - (1. / (N_pos * N_neg)) * dot;
			}

			//tX += ((clock() - (t1))*1.0f / CLOCKS_PER_SEC);
			return auc;
		}

		
		/*
		template<typename Tx>
		double AUC_cys_1(size_t dim, const Tx *label, const Tx *y1, int flag = 0x0) {	//失败的尝试，晕
			clock_t t1 = clock();
			//double auc_0 = AUC_Jonson(dim, label, y1, flag);
			size_t i, N_pos = 0, N_neg = 0,nz_1=0,nz_0=0,nz=0;
			Tx N_0 = y1[0], N_1 = y1[0], P_0 = y1[0], P_1 = y1[0],a;
			for (i = 0; i < dim; i++) {
				a = y1[i];
				if (label[i] == 1) {
					N_pos++;
					P_0 = min(P_0, a);		P_1 = max(P_1, a);
				}	else {
					N_0 = min(N_0, a);		N_1 = max(N_1, a);
				}
			}
			N_neg = dim - N_pos;
			if (N_pos == 0) {
				auc = 1;		return auc;
			}	else if (N_neg == 0) {
				auc = 1;		return auc;
			}

			double dot = 0, auc = 0;
			tpSAMP_ID *idx=new tpSAMP_ID[dim];
			for (nz_1 = 0, nz_0 = 0,i = 0; i < dim; i++) {
				//idx[nz++] = i;			continue;
				a = y1[i];
				if (label[i] == 0) {
					if (a < P_0) {
						nz_0++;
					}	else
						idx[nz++] = i;
				}	else {
					if (a > N_1) {
						dot += dim - nz_1;		nz_1++;
					}	else
						idx[nz++] = i;
				}
			}
			assert(nz+nz_0+nz_1==dim);
			//clock_t t1 = clock();
			Tx *y2 = new Tx[dim];
			memcpy(y2, y1, sizeof(Tx)*dim);
			std::sort(y2, y2 + dim);
			delete[] y2;
			std::sort(idx,idx+nz, [&y1](tpSAMP_ID i1, tpSAMP_ID i2) {return y1[i1] > y1[i2]; });
			//ParallelSort(idx.begin(), idx.end(), [&y1](tpSAMP_ID i1, tpSAMP_ID i2) {return y1[i1] > y1[i2]; }, IteratorValType(idx.begin()));
			//tX += ((clock() - (t1))*1.0f / CLOCKS_PER_SEC);
			for (i = 0; i < nz; i++) {		//首先对score从大到小排序，然后令最大score对应的sample 的rank为n
				if (label[idx[i]] == 0)
					continue;
				dot += (dim-nz_1)-i;
			}
			auc = (dot - (N_pos + 1)*N_pos / 2.0) / (N_pos*N_neg);
			//auc = 1 + ((N_pos + 1.) / (2 * N_neg)) - (1. / (N_pos * N_neg)) * dot;
			delete[] idx;
			tX += ((clock() - (t1))*1.0f / CLOCKS_PER_SEC);
			return auc;
		}*/


		template<typename Tx>
		double AUC_cys(size_t dim, const Tx *label, const Tx *y1, int flag = 0x0) {	//失败的尝试，晕
			clock_t t1 = clock();
			size_t i, nStep=1024,pos,*ptr=new size_t[nStep*2+2](),*count=ptr+ nStep+1, nz_1=0;
			double dot = 0, auc = 0;		
			StatAtLabel(dim, label, y1, flag);
			if (N_pos == 0) {
				auc = 1;		return auc;
			}
			else if (N_neg == 0) {
				auc = 1;		return auc;
			}
			//double auc_0= AUC_Jonson(dim,label,y1,flag)
			Tx a;
			if (P_1 == N_0 && P_1==N_1 && P_1==P_0) {
				for (i = 0; i < dim; i++) {
					if (label[i] == 0)
						continue;
					dot += dim-i;
				}
			}	else {
				double step = (P_1 - P_0) / (nStep - 1);		assert(step > 0);
				for (i = 0; i < dim; i++) {
					a = y1[i];
					if (a < P_0) {
						ptr[0]++;	 continue;
					}
					else if (a > P_1) {
						nz_1++;	continue;
					}
					pos=(int)((y1[i] - P_0)/step);
					count[pos]++;
				}
				for (i = 0; i < nStep; i++) {
					ptr[i + 1] = ptr[i] + count[i];
				}
				assert(ptr[nStep]+nz_1 == dim);		
				for (i = 0; i < dim; i++) {
					if (label[i] == 0)
						continue;
					pos = (int)((y1[i] - P_0) / step);
					dot += ptr[pos];
					ptr[pos]++;
				}

			}
			auc = (dot - (N_pos + 1)*N_pos / 2.0) / (N_pos*N_neg);
			//double auc_0 = AUC_Jonson(dim, label, y1, flag);
			delete[] ptr;
			//tX += ((clock() - (t1))*1.0f / CLOCKS_PER_SEC);
			return auc;
		}
	};

};


