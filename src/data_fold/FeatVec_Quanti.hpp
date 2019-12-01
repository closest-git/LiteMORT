#pragma once

#include "./DataFold.hpp"

namespace Grusoft {
	/*
		quantization
	*/
	template<typename tpQUANTI>
	class FeatVec_Q : public FeatVec_T<tpQUANTI> {
		//tpQUANTI binNA = tpQUANTI(-1);
	protected:
		FeatVector *hFeatSource = nullptr;
		//FeatBlit box;
		HistoGRAM *qHisto_0 = nullptr;
		//由PerturbeHisto生成，需要重新设计
		HistoGRAM *qHisto_1 = nullptr;
	public:
		FeatVec_Q(const FeatsOnFold *hData_, FeatVector *hFeat, int x, int flag = 0x0):hFeatSource(hFeat){
			FeatVector::id = hFeat->id;
			FeatVector::desc = hFeat->desc;
			FeatVector::nam = hFeat->nam;
			FeatVector::type = hFeat->type;
			FeatVector::hFold_ = hFeatSource->AtFold_();
			//FeatVector::hDistri = hFeatSource->hDistri;
			FeatVector::distri_ = hFeatSource->myDistri();
			FeatVec_T<tpQUANTI>::nSamp_0 = hFeatSource->size();
			FeatVector::PY = hFeatSource->PY;
		}

		virtual ~FeatVec_Q() {
			if (qHisto_0 != nullptr)			delete qHisto_0;
			if (qHisto_1 != nullptr)			delete qHisto_1;
			if (hFeatSource != nullptr) {
				delete hFeatSource;
				FeatVector::distri_ = nullptr;
			}
		}
		virtual HistoGRAM *GetHisto(int flag = 0x0) { return qHisto_0; }
		virtual const HistoGRAM *GetHisto(int flag = 0x0) const { return (const HistoGRAM *)qHisto_0; }

		void InitSampHisto(HistoGRAM* histo, bool isRandom, int flag = 0x0)	const {
			if (qHisto_0->nBins == 0) {
				histo->ReSet(0);	return;
			}
			else {
				histo->CopyBins(*qHisto_0, true, 0x0);
			}			
		}

		virtual void Observation_AtSamp(LiteBOM_Config config, SAMP_SET& samp, Distribution&distri, int flag = 0x0) {
			hFeatSource->Observation_AtSamp(config, samp, distri, flag);
		}

		/*
			static bin mapping		生成基于EDA的格子	参见Samp2Histo

			v0.1	cys
				10/22/2018
			v0.2	cys
				11/04/2019
		*/
		virtual void UpdateHisto(const FeatsOnFold *hData_, bool isOnY, bool isFirst, int flag = 0x0) {
			if (qHisto_0 != nullptr)
				delete qHisto_0;
			size_t nSamp = this->size(), i, samp, nValid = 0;
			if (nSamp != hData_->nSample()) {	//确实有可能 1 merge_right
				//assert(BIT_TEST(hFeat->type, FeatVector::RIGTH_MERGE) );
				const FeatsOnFold* src_fold_ = hFeatSource->AtFold_();
				assert(src_fold_->isMerge());
				i = 0;
			}
			string optimal = hData_->config.leaf_optimal;
			//qHisto = optimal == "grad_variance" ? new HistoGRAM(nSamp) : new Histo_CTQ(nSamp);
			qHisto_0 = new HistoGRAM(this, nSamp);
			//FeatVector *hFeat = hFeatSource;
			size_t nMostBin = hData_->config.feat_quanti;
			//bool isOnY = hData_->config.histo_bins_onY();	//hData_->config.histo_algorithm == LiteBOM_Config::HISTO_ALGORITHM::on_Y;
			tpDOWN *yDown = nullptr;		//明显不合理	3/11/2019
			if (isOnY) {
				if (isFirst) {
					//((FeatsOnFold *)hData_)->lossy->Update((FeatsOnFold *)hData_);
				}
				yDown = ((FeatsOnFold *)hData_)->GetDownDirection();
			}/**/
			 //assert(val == nullptr);
			if (FeatVec_T<tpQUANTI>::val == nullptr) {
				FeatVec_T<tpQUANTI>::val = new tpQUANTI[nSamp];
				BIT_RESET(FeatVector::type, FeatVector::VAL_REFER);
			}
			//val.resize(nSamp);
			tpQUANTI *quanti = FeatVec_T<tpQUANTI>::arr(), no;
			//qHisto->quanti = quanti;
			for (i = 0; i<nSamp; i++)	quanti[i] = tpQUANTI(-1);		//-1 for NAN

			ExploreDA *edaX = hData_->edaX;
			if (edaX != nullptr /*&& hData_->config.histo_algorithm == LiteBOM_Config::HISTO_ALGORITHM::on_EDA*/) {
				//Distribution& distri = edaX->arrDistri[FeatVector::id];
				Distribution* distri = edaX->GetDistri(FeatVector::id);
				if (isOnY) {
					throw "!!!histogram_bins onY is ...!!!";
					FeatVec_T<float> *hFeatFloat = dynamic_cast<FeatVec_T<float>*>(hFeatSource);
					float *x = hFeatFloat->arr();
					distri->ClearHisto();
					distri->X2Histo_(hData_->config, nSamp, x, yDown);
				}

				if (distri->histo != nullptr)
					qHisto_0->CopyBins(*(distri->histo), true, 0x0);
				hFeatSource->QuantiAtEDA(edaX,quanti,sizeof(tpQUANTI), nMostBin,  hData_, 0x0);
			}
			else {
				//hFeat->Split2Quanti(hData_->config,edaX, vThrsh, qHisto, yDown, nMostBin);
				printf("\n!!! FeatVec_Q::Update_Histo edaX=nullptr !!!\n");
				throw "\n!!! FeatVec_Q::Update_Histo edaX=nullptr !!!\n";
			}
			for (nValid = 0, i = 0; i < nSamp; i++) {
				/*if (quanti[i] == -111)		{
					printf("\n!!! FeatVec_Q::Update_Histo quanti[%d] is -111 !!!\n", i);		throw "\n!!! FeatVec_Q::Update_Histo quanti is -1 !!!\n";
				}*/
				if (quanti[i] >= 0)
					nValid++;
			}
			if (nValid == 0) {
				printf("\n FeatVec_Q(%s) nBin=%d a0=%g a1=%g", FeatVector::desc.c_str(), qHisto_0->nBins, 0.0, -1.0);
				BIT_SET(this->type, Distribution::DISTRI_OUTSIDE);
			}
			if (hData_->config.nMostSalp4bins>0 && hData_->isTrain())
				FeatVector::select_bins = new FS_gene_(this->nam, hData_->config.nMostSalp4bins, qHisto_0->nBins, 0x0);
			if (this->isCategory()) {

			}
			else {
				if (FeatVector::wBins != nullptr)
					delete[] FeatVector::wBins;
				FeatVector::wBins = new float[qHisto_0->nBins]();
				FeatVector::wSplit_last = 0;
			}

			//printf("\n FeatVec_Q(%s) nBin=%d a0=%g a1=%g", desc.c_str(),qHisto->bins.size(),qHisto->a0, qHisto->a1 );	
		}

		/*
			根据样本集，修正每个格子的内容(Y_sum,nz...)
			samps对应于quanti,	与samp_set.samps有微妙区别!!!
		*/
		virtual void Samp2Histo(const FeatsOnFold *hData_, const SAMP_SET&samp_set,  HistoGRAM* histo, int nMostBin,const tpSAMP_ID *samps4quanti =nullptr, int flag0 = 0x0)	const  {
			tpDOWN *hessian = hData_->GetSampleHessian();
			if (hessian == nullptr) {
				Samp2Histo_null_hessian(hData_, samp_set, histo, nMostBin, samps4quanti, flag0);
				//Samp2Histo_null_hessian_sparse(hData_, samp_set, histo, nMostBin, flag0);
			}	else {
				//histo->nSamp = samp_set.nSamp;
				const tpQUANTI *quanti = FeatVec_T<tpQUANTI>::arr();
				tpQUANTI no, *map = nullptr;
				InitSampHisto(histo, false);
				if (histo->nBins == 0) {
					return;
				}
				tpDOWN *down = hData_->GetSampleDown();
				string optimal = hData_->config.leaf_optimal;
				bool isLambda = optimal == "lambda_0";
				size_t nSamp = samp_set.nSamp, i, nSamp4 = 0;
				if (nSamp == hData_->nSample()) {
					hessian = hData_->GetHessian();
					down = hData_->GetDownDirection();
				}
				const tpSAMP_ID *samps = samp_set.samps;
				if (samps4quanti != nullptr)
					samps = samps4quanti;
				tpSAMP_ID samp;
				tpDOWN a;
				//histo->CopyBins(*qHisto, true, 0x0);
				int nBin = histo->nBins;//bins.size();
				HISTO_BIN *pBins = histo->bins, *pBin;	//https://stackoverflow.com/questions/7377773/how-can-i-get-a-pointer-to-the-first-element-in-an-stdvector
				{		//主要的时间瓶颈
					nSamp4 = 4 * (int)(nSamp / 4);
					for (i = 0; i < nSamp4; i += 4) {
						HISTO_BIN *pB0 = pBins + quanti[samps[i]];
						HISTO_BIN *pB1 = pBins + quanti[samps[i + 1]];
						HISTO_BIN *pB2 = pBins + quanti[samps[i + 2]];
						HISTO_BIN *pB3 = pBins + quanti[samps[i + 3]];
						tpDOWN a0 = down[i], a1 = down[i + 1], a2 = down[i + 2], a3 = down[i + 3];
						pB0->G_sum -= a0;			pB1->G_sum -= a1;			pB2->G_sum -= a2;			pB3->G_sum -= a3;
						pB0->H_sum += hessian[i];			pB1->H_sum += hessian[i + 1];
						pB2->H_sum += hessian[i + 2];			pB3->H_sum += hessian[i + 3];
						pB0->nz++;	pB1->nz++;	pB2->nz++;	pB3->nz++;
					}/**/
					 //if(nSamp<10000)
					for (i = nSamp4; i<nSamp; i++) {
						auto pos = quanti[samps[i]];
						assert(pos >= 0 && pos < nBin);

						//a = down[samp];
						a = down[i];
						pBin = pBins + pos;	//HISTO_BIN& bin = histo->bins[no];
						pBin->G_sum += -a;
						pBin->H_sum += hessian[i];
						//pBin->H_sum += hessian == nullptr ? 1 : hessian[samp];
						pBin->nz++;
					}
					//if(hParent==nullptr)
				}
				histo->CheckValid(hData_->config);
			}
#ifdef _DEBUG
			if (true /* && !isRandomDrop*/) {
				double G_sum = 0;	// histo->hBinNA()->G_sum;
				for (int i = 0; i < histo->nBins; i++) {
					//for (auto item : histo->bins) {
					G_sum += histo->bins[i].G_sum;
				}
				assert(fabs(G_sum + samp_set.Y_sum_1)<1.e-7*fabs(G_sum) || fabs(samp_set.Y_sum_1)<0.001);
			}
#endif
		}
			
		virtual size_t UniqueCount(const SAMP_SET&samp_set, int flag = 0x0) {
			size_t i, nSamp = samp_set.nSamp, nUnique;
			tpQUANTI *quanti = FeatVec_T<tpQUANTI>::arr(), no;
			const tpSAMP_ID *samps = samp_set.samps;
			set<int>mapu;
			for (i = 0; i < nSamp; i += 4) {
				no = quanti[samps[i]];
				mapu.insert(no);
			}
			nUnique = mapu.size();
			return nUnique;
		}

		virtual void Samp2Histo_null_hessian(const FeatsOnFold *hData_, const SAMP_SET&samp_set, HistoGRAM* histo, int nMostBin, const tpSAMP_ID *samps4quanti = nullptr, int flag = 0x0)	const {
			const HistoGRAM *qHisto = GetHisto();
			tpDOWN *down = hData_->GetSampleDown(),a;
			string optimal = hData_->config.leaf_optimal;
			bool isLambda = optimal == "lambda_0";
			size_t nSamp = samp_set.nSamp, i, nSamp_LD = 0, LD = 4;
			if (nSamp == hData_->nSample()) {
				down = hData_->GetDownDirection();
			}
			const tpSAMP_ID *samps = samp_set.samps;
			histo->CopyBins(*qHisto, true, 0x0);
			int nBin = histo->nBins;// bins.size();
			if (samps4quanti != nullptr) {
				samps = samps4quanti;				
			}
			tpSAMP_ID samp;
			const tpQUANTI *quanti = FeatVec_T<tpQUANTI>::arr();
			tpQUANTI no;
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
			histo->CheckValid(hData_->config);
		}

		//virtual void Samp2Histo_null_hessian_sparse(const FeatsOnFold *hData_, const SAMP_SET&samp_set, HistoGRAM* histo, int nMostBin, int flag = 0x0);

		//virtual void UpdateFruit(const FeatsOnFold *hData_, MT_BiSplit *hBlit, int flag = 0x0);

		friend class FeatVec_Bundle;
	};

}

