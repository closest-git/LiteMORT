#include "BiSplit.hpp"
#include "BoostingForest.hpp"
#include "../data_fold/DataFold.hpp"
#include "../data_fold/FeatVec_2D.hpp"
#include "../util/Object.hpp"

using namespace Grusoft;

double MT_BiSplit::tX = 0;

FRUIT::FRUIT(HistoGRAM *his_, int flag) : histo(his_) {
	if( histo!=nullptr)
		histo->fruit=this;
}

FRUIT::~FRUIT() {
	if (histo != nullptr)
		delete histo;
	//if (bsfold != nullptr)
	//	delete bsfold;
}

/*
	v0.2		cys	
		3/29/2019
*/
void SAMP_SET::SampleFrom(FeatsOnFold *hData_, const SAMP_SET *from, size_t nMost, int rnd_seed, int flag) {
	Alloc(nMost);
	size_t i, nFrom = hData_->nSample(),nz=0,pos;
	if (from == nullptr) {
	}	else {
		assert(from != nullptr && nMost<from->nSamp);
		nFrom = from->nSamp;
	}
	if (nMost >= nFrom) {
		for (size_t i = 0; i < nMost; i++) {
			root_set[i] = i;
		}
		//memcpy(root_set, from->samps,sizeof(tpSAMP_ID)*nFrom);
		return;
	}
	size_t T_1 = nFrom / std::log2(nMost*1.0);
	unsigned int x = 123456789,next; 
	//srand(time(0)); 
	x = hData_->rander_samp.RandInt32() % nFrom;
	bool isSequence = true && from == nullptr;
	if (isSequence) {		//得失之间...		4/11/2019	cys
		/*size_t n_X = min(nFrom - x, nMost),n_2 = n_X >=nMost ? 0 : nMost- n_X;
		for (i = 0; i < n_2; i++) {
			root_set[i] = i ;
		}
		for (i = 0; i < n_X; i++) {
			root_set[n_2+i] = i+x;
		}*/
		size_t grid = max(int(nMost/100), 1);
		while (nz < nMost) {	
			size_t start = hData_->rander_samp.RandInt32() % nFrom,end=min(grid, nFrom-start);
			end = min(end, nMost - nz);
			end += start;
			for (i = start; i < end; i++) {
				root_set[nz++] = i;
			}
		}
		//std::sort(root_set, root_set + nMost);
	}else	if (nMost > T_1) {
		for (nz=0, i = 0; i < nFrom; ++i) {
			double prob = (nMost - nz) / static_cast<double>(nFrom - i);
			x = (214013 * x + 2531011);
			x = ((x >> 16) & 0x7FFF);
			if ((x/32768.0f) < prob) {
				root_set[nz++]=i;
			}
		}
	}	else {
		tpSAMP_ID *mask=new tpSAMP_ID[nFrom]();
		while (nz < nMost) {
			x = (214013 * x + 2531011);
			//static_cast<int>(x & 0x7FFFFFFF);
			next = x % nMost;
			if (mask[next] == 0) {
				root_set[nz++] = next;
				mask[next] = 1;
				//sample_set.insert(next);
			}
		}	
		delete[] mask;
		std::sort(root_set, root_set + nMost);
	}
	assert(nz <= nMost);
}

MT_BiSplit::MT_BiSplit(FeatsOnFold *hData_, int d, int rnd_seed, int flag) : depth(d) {
	assert(hData_ != nullptr);
	double subsample= hData_->config.subsample;
	
	size_t i, nSamp = hData_->nSample();	
	if (subsample < 0.999) {
		size_t nMost= nSamp*subsample;
		//samp_set.SampleFrom(hData_,&(hData_->samp_set),nMost, rnd_seed);		
		samp_set.SampleFrom(hData_, nullptr, nMost, rnd_seed);
	}	else {
		//samp_set.SampleFrom(hData_, &(hData_->samp_set), nSamp, rnd_seed);
		samp_set.SampleFrom(hData_, nullptr, nSamp, rnd_seed);
		//samp_set = hData_->samp_set;
		//samp_set.isRef = true;
	}
	Observation_AtLocalSamp(hData_);
	Init_BFold(hData_);
	hData_->stat.dY = samp_set.Y_1- samp_set.Y_0;
}

/*
	样本只是某个泛函的观测值!!!
	gain,imputiry,down必须保持一致

	v0.1	cys
		9/4/2018
	v0.2	cys
		1/26/2018

*/
void MT_BiSplit::Observation_AtLocalSamp(FeatsOnFold *hData_, int flag) {
	char temp[2000];
	string optimal = hData_->config.leaf_optimal;

	impuri = 0;		devia = 0;
	size_t dim = nSample();
	if (dim == 0)
		return;
	tpDOWN *down = hData_->GetDownDirection(),*hess= hData_->GetHessian();
	//double a, x_0 = DBL_MAX, x_1 = -DBL_MAX;
	tpDOWN a, a2 = 0.0, mean = 0, y_0, y_1;
	double DOWN_sum = 0;
	samp_set.STA_at<tpDOWN>(down, a2, DOWN_sum, y_0, y_1,true);
	mean = DOWN_sum / dim;
	G_sum = -DOWN_sum;		//很重要 gradient方向和down正好相反
	Y2_sum = a2;
	//if (y_0 == y_1 || fabs(y_0-y_1)<1.0e-6*fabs(y_0)) {	//基本为常量
	if ZERO_DEVIA(y_0,y_1) {	//基本为常量
	}	else {
		//mean /= dim;
		a = a2 - dim*mean*mean;
		impuri = (double)(a);
		if (impuri<0 && fabs(impuri)<1.0e-6*a2)
			impuri = 0;
		//if (impuri == 0)					assert(0);
		assert(impuri >= 0);
		devia = sqrt(impuri / dim);		
	}
//REF:	"XGBoost: A Scalable Tree Boosting System" 对于mse-loss 基本等价
	/*if (optimal == "taylor_2") {	
		H_sum = 2*dim;		G_sum=-2*mean*dim;
		impuri = G_sum*G_sum/H_sum;
		//down_step = mean/2;
		down_step = -G_sum/ H_sum;
	}	else*/ 
	if (optimal == "lambda_0") {
		if (hess == nullptr) {
			H_sum = dim;
		}	else {
			double h2, h_0, h_1;
			samp_set.STA_at<tpDOWN>(hess, h2, H_sum, h_0, h_1,false);
		}
		impuri = G_sum*G_sum / H_sum;		//已略去常数
		down_step = -G_sum / H_sum;
		sprintf(temp, "impuri(%g/%g %d)", G_sum, H_sum, dim);
		sX = temp;
	}else	{
		down_step = mean;
		//down_step = sqrt(a2/dim);		 为啥这样不行，有意思
	}
	//printf("%.4g ", down_step);
	double shrink = hData_->config.learning_rate;
	//double init_score=hData_->lossy.init_score;
	down_step = down_step*shrink;
	return;
}

double MT_BiSplit::GetGain(int flag) {
	return 0x0;
}

//由于浮点误差的原因，很小的负数也是0
double FLOAT_ZERO(double a,double ruler) {
	if (a > -DBL_EPSILON * 1000 * ruler && a < 0)	//Y2_sum,mxmxN计算方式不一样，确实会有浮点误差
		return 0;
	return a;
}

/*
	累进特征的展开
		例如消费者一个时间段的消费记录
*/
double MT_BiSplit::AGG_CheckGain(FeatsOnFold *hData_, FeatVector *hFeat, int flag) {
	assert(BIT_TEST(hFeat->type, FeatVector::AGGREGATE));
	FRUIT *fruit = nullptr;
	HistoGRAM *histo = fruit->histo;
	FeatVector *hAF = nullptr;
	int nExpand = 100, i, exp_no =-1;
	double mxmxN = 0;
	for(i=0;i<nExpand;i++){
		hFeat->Samp2Histo(hData_, samp_set, histo, hData_->config.feat_quanti);
		histo->GreedySplit_X(hData_, samp_set);
		if (mxmxN < fruit->mxmxN) {
			mxmxN = fruit->mxmxN, exp_no = i;
		}
	}
	hAF->agg_no = exp_no;
	return 0;
}

/*
	v0.1	cys
		2/26/2019
*/
int MT_BiSplit::PickOnGain(FeatsOnFold *hData_,const vector<FRUIT *>& arrFruit, int flag) {
	double mxmxN = 0;
	fruit = nullptr;
	vector<double> cands;
	int pick_id=-1, nFruit = arrFruit.size(),i;
	bool isRanomPick = false;	//hData_->config.random_feat_on_gain
	for (i = 0; i<nFruit; i++) {		//为了并行
		FRUIT *fr_ = arrFruit[i];
		if (fr_ == nullptr || fr_->mxmxN <= 0)
		{			cands.push_back(-1);	continue;		}
		cands.push_back(fr_->mxmxN-impuri);		//BUG 需要减去impuri
		if (mxmxN < fr_->mxmxN) {
			mxmxN = fr_->mxmxN, pick_id = i;
			//feat_id = picks[pick_id];		feat_regress = -1;
			//fruit = fr_;
		}
	}
	double T = (mxmxN - impuri)*0.95;
	if (isRanomPick && cands.size()>10) {
		assert(cands.size() == arrFruit.size());
		vector<size_t> idx_1;
		idx_1.resize(cands.size());// initialize original index locations
		iota(idx_1.begin(), idx_1.end(), 0);
		// sort indexes based on comparing values in v
		std::sort(idx_1.begin(), idx_1.end(), [&cands](size_t i1, size_t i2) {return cands[i1] > cands[i2]; });
		assert(mxmxN == cands[idx_1[0]]);
		size_t range = max(1, (int)(cands.size() / 10)), pos = -1;
		for (i = 0; i < cands.size() - 1; i++) {
			assert(cands[idx_1[i]] >= cands[idx_1[i + 1]]);
			if (cands[idx_1[i]] > T) {
				range = i + 1;
			}
		}
		pos = rand() % range;
		pick_id = idx_1[pos];		
	}
	return pick_id;
}

/*
	v0.2	并行
*/
double MT_BiSplit::CheckGain(FeatsOnFold *hData_, const vector<int> &pick_feats, int x, int flag) {
	GST_TIC(t1)
	if (bsfold != nullptr) {
		bsfold->GreedySplit(hData_, flag);
		//fruit = new FRUIT(bsfold);
		return gain;
	}
	if (this->id == 56) {
		int i = 0;		//仅用于调试
	}
	/*if (samp_set.Y_1 - samp_set.Y_0 < hData_->stat.dY / 10) {
		printf( "\n!!!Tiny Y:::just PASS!!!	Tree=%d node=%d, samp_set=<%g-%g> |y|=%g", hModel->skdu.noT,this->id,
			samp_set.Y_0,samp_set.Y_1, hData_->stat.dY );
		impuri = 0;
		return 0;
	}*/
	int nThread = hData_->config.num_threads,pick_id=-1,node_task= hData_->config.node_task;
	size_t nSamp = samp_set.nSamp,i,step= pick_feats.size();
	string optimal = hData_->config.leaf_optimal;
	assert(impuri>0);
	assert(nSamp >= hData_->config.min_data_in_leaf);
	//double bst_split=0;
	vector<int> picks= pick_feats;
	//hData_->nPick4Split(picks, hData_->rander_feat);		//似乎效果一般，奇怪	3/7/2019		cys
	feat_id=-1;
//	picks.resize(1);		//仅用于测试
	//if (task == "split_X" || task == "split_Y")
	if (node_task == LiteBOM_Config::split_X || node_task == LiteBOM_Config::histo_X_split_G)
		assert(gain == 0);
	vector<FRUIT *> arrFruit;
	arrFruit.resize(picks.size());
	//fBlits.resize(picks.size());
	tpDOWN *yDown = hData_->GetDownDirection();
	//picks.clear();		picks.push_back(24);		//仅用于调试
	int num_threads = OMP_FOR_STATIC_1(picks.size(), step);
	if (false) {
		BinFold bf(hData_,picks, samp_set);
		//bf.GreedySplit(hData_, picks ,0x0 );
	}

	size_t start = 0, end = picks.size();
#pragma omp parallel for num_threads(nThread) schedule(dynamic)
	for (int i = start; i < end; i++) {
		int pick = picks[i];
		if (i == 24 && this->id == 7) {	//仅用于测试
			//i = 24;
		}
		FeatVector *hFeat = hData_->Feat(pick);
		//HistoGRAM *histo = optimal=="grad_variance" ? new HistoGRAM(nSamp) : new Histo_CTQ(nSamp);
		HistoGRAM *histo = new HistoGRAM(hFeat, nSamp);
		arrFruit[i] = new FRUIT(histo);
		if (nSamp == 400 && pick == 9)
			pick = 9;

		if (BIT_TEST(hFeat->type, FeatVector::AGGREGATE)) {
			AGG_CheckGain(hData_, hFeat, flag);
		}
		else {
			hFeat->Samp2Histo(hData_, samp_set, histo, hData_->config.feat_quanti);		//Histo Normalize
			if (histo->bins.size() == 0)
				continue;
			if (hFeat->isCategory()) {
				histo->GreedySplit_Y(hData_, samp_set, false);
				//continue;
			}
			else {
				if (node_task == LiteBOM_Config::split_X)
					histo->GreedySplit_X(hData_, samp_set);
				else if (node_task == LiteBOM_Config::histo_X_split_G)
					histo->GreedySplit_Y(hData_, samp_set, true);
				else if (node_task == LiteBOM_Config::REGRESS_X)
					histo->Regress(hData_, samp_set);
				else
					throw "MT_BiSplit::CheckGain task is !!!";
			}
		}
		//节省histo所占的空间,	如有需要，可重新调用Samp2Histo
		if (!arrFruit[i]->isY) {
			delete histo;		arrFruit[i]->histo = nullptr;
		}
	}


	/*for (int i = 0; i < 10; i++) {		//2d feat vector
		//FeatVec_2D *hFQ = new FeatVec_2D(hData_, 0, f2D,0);
		HistoGRAM_2D *histo = new HistoGRAM_2D(nSamp);
		FRUIT *f2D = new FRUIT(histo);
		histo->GreedySplit_X(hData_, samp_set, minSet);
	}*/
	double mxmxN = 0;
	pick_id = PickOnGain(hData_, arrFruit, flag);	
	if (pick_id >= 0) {
		feat_id = picks[pick_id];		feat_regress = -1;
		fruit = arrFruit[pick_id];
		mxmxN = fruit->mxmxN;
		FeatVector *hFeat = hData_->Feat(picks[pick_id]);	
		hFeat->UpdateFruit(hData_,this);
		if (hFeat->hDistri != nullptr && hFeat->hDistri->rNA > 0) {			
			//fruit->isNanaLeft = hFeat->hDistri;
		}	
	}
	else {	//确实有可能,很多情况可以进一步优化，需要积累一些算例来分析
		//printf("\n\t!!! Failed split at %d nSamp=%d nPick=%d !!!\n", id, nSample(), picks.size());
	}
	for (int i = 0; i<picks.size(); i++) {		//为了并行
		if( fruit == arrFruit[i])
		{	continue;		}
		else
			delete arrFruit[i];
	}
	arrFruit.clear( );
	hData_->stat.nCheckGain++;
	if (fruit != nullptr) {
		if (optimal == "lambda_0") {
			gain = FLOAT_ZERO(mxmxN-impuri, mxmxN);
			if (gain < 0) {
				//Observation_AtSamp(hData_);
				assert(gain>=0);
			}
			//return gMax;
		}	else {
			double bst_imp = Y2_sum - mxmxN;
			bst_imp = FLOAT_ZERO(Y2_sum - mxmxN, mxmxN);
			//if (bst_imp > -DBL_EPSILON*1000* Y2_sum && bst_imp < 0)	//Y2_sum,mxmxN计算方式不一样，确实会有浮点误差
			//	bst_imp = 0;
			gain = impuri - bst_imp;		assert(gain>=0);			
			if (!(bst_imp >= 0 && bst_imp < impuri)) {
				printf("\n!!!! bst_imp=%5.3g impuri=%5.3g Y_sum_2=%5.3g mean*mean*N=%5.3g!!!!", bst_imp, impuri, Y2_sum, mxmxN);
				//assert(0);
			}
		}
	}
	FeatsOnFold::stat.tX += GST_TOC(t1);
	return gain;
}

void MT_BiSplit::Init_BFold(FeatsOnFold *hData_, int flag) {

}