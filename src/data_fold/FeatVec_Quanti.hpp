#pragma once

#include "./DataFold.hpp"

namespace Grusoft {
	/*
		quantization
	*/
	//template<typename tpQUANTI>
	class FeatVec_Q : public FeatVec_T<tpQUANTI> {
	protected:
		FeatVector *hFeatSource = nullptr;
		//FeatBlit box;
		HistoGRAM *qHisto_0 = nullptr;
		//由PerturbeHisto生成，需要重新设计
		HistoGRAM *qHisto_1 = nullptr;
	public:
		FeatVec_Q(const FeatsOnFold *hData_, FeatVector *hFeat, int nMostBin, int flag = 0x0);
		virtual ~FeatVec_Q() {
			if (qHisto_0 != nullptr)			delete qHisto_0;
			if (qHisto_1 != nullptr)			delete qHisto_1;
			if (hFeatSource != nullptr) {
				delete hFeatSource;
				hDistri = nullptr;
			}
		}
		HistoGRAM *GetHisto(int flag = 0x0) { return qHisto_0; }

		void InitSampHisto(HistoGRAM* histo, bool isRandom, int flag = 0x0);

		virtual void Observation_AtSamp(LiteBOM_Config config, SAMP_SET& samp, Distribution&distri, int flag = 0x0) {
			hFeatSource->Observation_AtSamp(config, samp, distri, flag);
		}

		//virtual tpQUANTI *GetQuantiBins(int flag = 0x0) { 			return val; 		}

		//static bin mapping	生成基于EDA的格子	参见Samp2Histo
		//virtual void PerturbeHisto(const FeatsOnFold *hData_, int flag = 0x0);
		virtual void UpdateHisto(const FeatsOnFold *hData_, bool isOnY, bool isFirst, int flag = 0x0);
		//根据样本集，修正每个格子的内容(Y_sum,nz...)
		virtual void Samp2Histo(const FeatsOnFold *hData_, const SAMP_SET&samp_set, HistoGRAM* hParent, HistoGRAM* histo, int nMostBin, int flag = 0x0);
		virtual size_t UniqueCount(const SAMP_SET&samp_set, int flag = 0x0);
		virtual void Samp2Histo_null_hessian(const FeatsOnFold *hData_, const SAMP_SET&samp_set, HistoGRAM* histo, int nMostBin, int flag = 0x0) {
			HistoGRAM *qHisto = GetHisto();
			tpDOWN *down = hData_->GetSampleDown();
			string optimal = hData_->config.leaf_optimal;
			bool isLambda = optimal == "lambda_0";
			size_t nSamp = samp_set.nSamp, i, nSamp_LD = 0, LD = 4;
			if (nSamp == hData_->nSample()) {
				down = hData_->GetDownDirection();
			}
			const tpSAMP_ID *samps = samp_set.samps;
			tpSAMP_ID samp;
			tpDOWN a;
			tpQUANTI *quanti = arr(), no;
			histo->CopyBins(*qHisto, true, 0x0);
			int nBin = histo->nBins;// bins.size();
			HISTO_BIN *pBins = histo->bins, *pBin;	//https://stackoverflow.com/questions/7377773/how-can-i-get-a-pointer-to-the-first-element-in-an-stdvector
			GST_TIC(t1);
			nSamp_LD = LD == 0 ? 0 : LD * (int)(nSamp / LD);
			for (i = 0; i < nSamp_LD; i += LD) {
				const auto p0 = quanti[samps[i]], p1 = quanti[samps[i + 1]], p2 = quanti[samps[i + 2]], p3 = quanti[samps[i + 3]];
				pBins[p0].G_sum -= down[i];
				pBins[p1].G_sum -= down[i + 1];
				pBins[p2].G_sum -= down[i + 2];
				pBins[p3].G_sum -= down[i + 3];
				++pBins[p0].nz;	++pBins[p1].nz;	++pBins[p2].nz;	++pBins[p3].nz;
				/*TO_BIN_01(pBins, quanti, samps++, down++);
				TO_BIN_01(pBins, quanti, samps++, down++);
				TO_BIN_01(pBins, quanti, samps++, down++);
				TO_BIN_01(pBins, quanti, samps++, down++);	*/
			}
			//if(nSamp<10000)
			for (i = nSamp_LD; i<nSamp; i++) {
				const auto p0 = quanti[samps[i]];
				pBins[p0].G_sum -= down[i];			++pBins[p0].nz;
				//TO_BIN_01(pBins, quanti, samps++, down++);
			}

			for (i = 0; i < nBin; i++) {
				pBins[i].H_sum = pBins[i].nz;
			}
		}
		virtual void Samp2Histo_null_hessian_sparse(const FeatsOnFold *hData_, const SAMP_SET&samp_set, HistoGRAM* histo, int nMostBin, int flag = 0x0);

		//virtual void UpdateFruit(const FeatsOnFold *hData_, MT_BiSplit *hBlit, int flag = 0x0);

		friend class FeatVec_Bundle;
	};

}

