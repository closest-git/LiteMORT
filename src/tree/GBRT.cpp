#include <omp.h>
#include "GBRT.hpp"
#include "../data_fold/Loss.hpp"
#include "../util/Object.hpp"
#include "../EDA/SA_salp.hpp"
#include "../learn/Pruning.hpp"


GBRT::GBRT(FeatsOnFold *hTrain, FeatsOnFold *hEval, double sOOB, MODEL mod_, int nTre_, int flag) : BoostingForest() {
	double rou = 1.0;

	model = mod_;	
	skdu.cascad = 0;			//skdu.nStep = nStep_;
	int seed = skdu.cascad + 31415927, nTrain = hTrain->nSample(), i, cls, nMostThread;
	skdu.nTree = nTree = rounds = nTre_;

	hTrainData = hTrain;	// new FeatsOnFold(nullptr, nTrain, 0, nCls, 0);
	hEvalData = hEval;
	//	hTrainData=hIn;		hTestData=hTes_;		
	size_t nSamp = hTrainData->nSample( );
	//InitRander(seed);
	
	//lenda = 10;	//MAX2(1.0,10-cas*0.5);			
#pragma omp parallel
	nMostThread = omp_get_num_threads();
	//eta = 0.1;
	//nBlitThread = 8;	//并行导致结果不再可重复
	
	maxDepth = hTrain->config.max_depth;
	//int histo = hTrain->config.histo_bins;
	nThread = hTrain->config.num_threads<=0 ? nMostThread : hTrain->config.num_threads;
	assert(nThread>0 && nThread<32);
	omp_set_num_threads(nThread);
	omp_set_nested(0);
#pragma omp parallel
	nThread = omp_get_num_threads();

	sBalance = 1.0;
	//min_set = MAX2(min_set, nSamp / 1000);		hTrain->config.min_data_in_leaf = min_set;
	nPickWeak = INT_MAX;
	if (regular == MULTI_TREE) {		//multi_tree改变了很多
		//eta = 1;		//需要adaptive learning rate,参见multi_1_23_2016.dat
						//maxDepth=5;	//其dat文件较4增加一倍
						//lenda=3.0;
	}
	else {
	}

	InitFeat(0x0);
	//nClass = nCls;
	nClass = hTrainData->nCls;
	for (i = 0; i<nTrain; i++) {
		/*double *Y= hTrain->Y(i);
		CASE *hX = new CASE(Y[0],0.5);
		SamplSet.push_back(hX);	*/	
	}
	nOOB = nTrain*sOOB;
	histo_buffer = new HistoGRAM_BUFFER(hTrain);
	hPruneData = hTrainData;	
	if (hEval != nullptr) {
		hPruneData = hEvalData;
	}
	if(hTrain->config.nMostPrune>0)
		prune = new EnsemblePruning(this, hPruneData, hTrain->config.nMostPrune);

	const char *mod = model==CLASIFY ? "CLASIFY" : "REGRESSION";
	printf("\n\n********* GBRT[%s]\n\tnTrainSamp=%d,nTree=%d,maxDepth=%d regress@LEAF=%s thread=%d feat_quanti=%d...",
		mod,nTrain, nTree, maxDepth, hTrain->config.leaf_regression.c_str(),nThread, hTrain->config.feat_quanti);
	hTrain->config.dump( );
	hTrain->present.dump();
	printf("\n********* GBRT *********\n" );
	//printf("\n\tlr=%g sample=%g leaf_optimal=\"%s\" num_leaves=%d*********\n********* GBRT *********\n",
	//	lr, hTrain->config.leaf_optimal.c_str(),hTrain->config.num_leaves);
}

void GBRT::BeforeTrain(FeatsOnFold *hData_, int flag ) {
	hData_->BeforeTrain( this,flag );
	if (0) {	//均值平移，似乎没有必要
		/*mSum /= nSample;	a = nSample*mSum.squaredNorm();
		for each(ShapeBMPfold *hFold in Trains) {
		hFold->vS -= mSum;
		hFold->UpdateErr();//vR=hFold->vS-hFold->vY;
		im1 += hFold->vR.squaredNorm();
		}*/
		//assert(fabs(im0 - a - im1)<FLT_EPSILON*im0);
	}
}


/*
	如果y已知，就可以checkLossy
	v0.1	集成hMove加速
	v0.2	root->samp_set允许非空
*/
double GBRT::Predict(FeatsOnFold *hData_, bool updateStopping,bool checkLossy, bool resumeLast, int flag) {
	size_t nSamp = hData_->nSample(), t;
	bool isEval = hData_->isEval();
	hData_->BeforePredict( );
	ManifoldTree *lastTree = nullptr;
	bool isResetZero=false;		//init_score带来了很多问题，需要重新设计,参见ManifoldTree::Train之AddScore
	if (resumeLast && forest.size()>0) {
		lastTree = (ManifoldTree*)forest[forest.size() - 1];
	}else{
		hData_->GetPrecict()->Empty();
		if (hData_->init_score.fVec !=nullptr) {
			hData_->GetPrecict()->CopyFrom(hData_->init_score.fVec);
		}	else if (forest.size() == 0) {
			if (hData_->init_score.step != 0) {
				hData_->GetPrecict()->Set(hData_->init_score.step);
			}
			isResetZero=true;
		}
	}
	FeatVec_T<tpDOWN>* predict = dynamic_cast<FeatVec_T<tpDOWN>*>(hData_->GetPrecict());
	assert(predict!=nullptr);
	tpDOWN *allx = predict->arr();

	GST_TIC(t1);
	for (auto tree : forest) {
		ManifoldTree *hTree = dynamic_cast<ManifoldTree *>(tree);
		assert(hTree!=nullptr);
		if (lastTree != nullptr && hTree != lastTree)
			continue;
		//if (forest.size() == 9 && nSamp == 1250)
		//	t = 0;

		bool isNodeMajor = true;
		if (true) {		//data-major	似乎快一些
			if (hTree->ArrTree_quanti !=nullptr) {
				/*if (isEval) {		// 需要快速评估函数， auc不合适啊
					hData_->UpdateStepOnReduce<float, double>(hTree->ArrTree_data, hTree->ArrTree_quanti);
				}*/
				if (hData_->isQuanti) {
					isNodeMajor = !hData_->PredictOnTree<double>(*(hTree->ArrTree_quanti), flag);
				}		else				{
					isNodeMajor = !hData_->PredictOnTree<double>(*(hTree->ArrTree_data), flag);					
				}// 
			}
		}
		if(isNodeMajor) {			//node-major
			hMTNode root = hTree->hRoot();
			bool isSplit =  root->nSample() == 0;
			if (hData_->hMove != nullptr)
				;// hData_->hMove->BeforeStep(hData_->samp_set, allx, 0x0);
			if (isSplit) {
				//root->samp_set = hData_->samp_set;				root->samp_set.isRef = true;
				root->samp_set.SampleFrom(hData_,this,nullptr, hData_->nSample(),-1);
			}
			for (auto node : hTree->nodes)	{
			//for each(hMTNode node in hTree->nodes) {
				//if( skdu.noT==31 && node->id==53 && BIT_TEST(hData_->dType, FeatsOnFold::DF_EVAL))
				//	node->id=53;
				if (node->isLeaf()) {
					hData_->AtLeaf(node, 0x0);
				}
				else {
					if (isSplit)				{
						hData_->SplitOn(node);
					}
				}
			}
			hTree->ClearSampSet();

		}
		if (hData_->hMove != nullptr);// hData_->hMove->AfterStep(hData_->samp_set, allx);
		
	}
	if(checkLossy)
		hData_->lossy->Update(hData_,this->skdu.noT,0x0);
	double err=DBL_MAX, y2 = DBL_MAX;
	if (checkLossy) {
		//on the objective
		if (hData_->config.objective == "binary") {
		}
		else {
		}
		err = hData_->lossy->ERR();	
		if (BIT_TEST(hData_->dType, FeatsOnFold::DF_EVAL)) {
			if ((skdu.noT <= 100 && skdu.noT % 5 == 0) || skdu.noT % hData_->config.verbose_eval == 0) {
				if (hData_->config.eval_metric == "auc") {
					printf("auc_%d=%-8.5g ", skdu.noT, hData_->lossy->err_auc);
				}else
					printf("%s_%d=%-8.5g ", hData_->nam.c_str(),skdu.noT,  err);		//eval_
				if(skdu.noT > 100)	printf("tX=%.3g ", FeatsOnFold::stat.tX);
			}

		}
		if (updateStopping) {
			bool isLRjump = false;
			stopping.Add(err, forest.size(), isLRjump);
		}		
	}
	if (isResetZero) {
		//if (hData_->init_score.fVec == nullptr)
		hData_->GetPrecict()->Empty();
	}
	else {
		/*tpSAMP_ID *samps = hData_->samp_set.samps;
		for (t = 0; t < nSamp; t++) {
			samps[t] = t;
		}*/
	}


	return err;
}

void  EARLY_STOPPING::CheckBrae(int flag) {
	nBraeStep = 0;
	if (errors.size() < early_round)
		return;
	if (best_no < errors.size() - 20) {
	//if (best_no < errors.size() - early_round/2) {
		nBraeStep = errors.size()- best_no;
	}
}

void EARLY_STOPPING::Add(double err,int best_tree, bool& isLRjump, int flag) {
	nLeastOsci = max(1, early_round / 20);
	isLRjump = false;
	errors.push_back(err);
	if (err <= e_best) {
		e_best=err;		
		best_no=errors.size()-1;
		best_round = best_tree;		
	}
	else {
		if (isOscillate==false && best_no+ nLeastOsci <= errors.size()- 1 ) {	//first isOscillate
			printf("\n-------- Oscillate@(%d,%g) best=(%d,%g) -------- \n", errors.size(), err, best_no+1, e_best);
			assert(err >= e_best);
			isOscillate = true;
		}
		if (errors.size() - best_no > early_round / 10 && LR_jump > 0) {
			printf("\n********* stopping JUMP e_best=%.6g@%d,cur=%.6g\t*********", e_best, best_no, err);
			best_no = errors.size()-1;		e_best = err;	//有问题
			LR_jump--;
			isLRjump = true;
		}
	}
}

/*
bool EARLY_STOPPING::isOscillate(int nLast) {
	if (isOsci_)
		return true;
	double e_last = errors[errors.size() - 1];
	if (best_no <= errors.size() - nLast-1) {
		printf("\n======Oscillate@%d\n", errors.size());
		assert(e_last >= e_best);
		isOsci_ = true;
		return true;
	}
	return false;
}*/

void EARLY_STOPPING::Reset() {
	*this = EARLY_STOPPING(early_round);
}

bool EARLY_STOPPING::isOK(int cur_round) {
	double e_last = errors[errors.size() - 1];
	if (true) {
		if( errors.size()<=early_round && e_best>0)
			return false;
		if (best_no<= errors.size()-1- early_round) {
			assert(e_last>=e_best);
			return true;
		}
	}else {
		if (cur_round<early_round)
			return false;
		//double e_last = errors[errors.size()-1];
		if (best_round <= cur_round - early_round) {
			assert(e_last >= e_best);
			return true;
		}

	}
	return false;
}

int GBRT::IterTrain(int round, int flag) {
	GST_TIC(tick);
	ManifoldTree *hTree = forest.size() == 0 ? nullptr : 
		dynamic_cast<ManifoldTree *>(forest[forest.size() - 1]);


	size_t nPickSamp = 0;
	int nIns = 0, no = 0, total, i, j, nzNode = 0, nIter = 0;
	double err_0 = DBL_MAX, err = DBL_MAX, a;
	bool isEvalTrain = true,isLRjump=false;
	//vector<double> err_eval;
	do {
		if (isEvalTrain) {
			FeatVector *hY1 = hTrainData->GetPrecict();
			tpDOWN *hDown = hTrainData->GetDownDirection();
			err_0 = this->Predict(hTrainData, hEvalData == nullptr, true, true);		//可以继续优化
			if (hTrainData->lossy->isOK(0x0, FLT_EPSILON)) {
				eOOB = 0;	printf("\n********* ERR@Train is ZERO, Break!!! *********\n\n");	return 0x0;
			}
			if (skdu.noT % 500 == 0 && hTrainData->config.verbose == 666) {
				a = forest.size() == 0 ? 0 : nzNode*1.0 / forest.size();
				printf("\n====== %d: ERR@%s=%8.5g nNode=%g nPick=[%d,%lld] time=%.3g======\n", skdu.noT, hTrainData->nam.c_str(), err_0,
					a, hTrainData->nPickFeat, nPickSamp, GST_TOC(tick));
			}
			//if (hEvalData == nullptr)
			//	stopping.Add(err_0, round, isLRjump);
		}
		if (hEvalData != nullptr) {
			if (round > 0) {
				hMTNode hRoot = (dynamic_cast<ManifoldTree*>(forest[round - 1]))->hRoot();		//im1 = hRoot->impuri;
				assert(hRoot->nSample() == 0);
			}
			err = this->Predict(hEvalData, true, true, true);	//经过校验，同样可以用resumeLast
			if (nIter > 0) {
				double err_last = stopping.curERR();
				if (err >= err_last)
					break;
				if (fabs(err_last - err) < err_last / 1000)
					break;
			}
			//stopping.Add(err, round,isLRjump);
			if (hEvalData->lossy->isOK(0x0, FLT_EPSILON)) {
				eOOB = 0;	printf("\n********* You are so LUCKY!!! *********\n\n");	return 0x0;
			}
		}
		nIter = nIter + 1;
	}	while (true);
	if (nIter > 1) {
		hTree->iter_refine = nIter;
	}
	return nIter;
}

int GBRT::Train(string sTitle, int x, int flag) {
	GST_TIC(tick);	
	eOOB = 1.0;
	size_t nSamp = nSample(), t;
	assert(forest.size() == 0);
	mSum = 0;	// mSum.setZero();	
	HistoGRAM::nAlloc = 0;
	//nzWeak=0.0;
	//参见BoostingForest::Train();
	stage = RF_TRAIN;
	int nIns = 0, no = 0, total, i, j, nzNode=0, maxDepth=0;
	//FeatsOnFold *hData=curF[0]->hData;
	total = hTrainData->nSample();			
	stopping.early_round = hTrainData->config.early_stopping_round;
	//hTrainData->feat_salps = new FS_gene_("select feature",64, hTrainData->feats.size(), 0x0);
	float *distri = hTrainData->distri, *dtr = nullptr, tag, d1, rOK = 0;
	double err_0= DBL_MAX,err=DBL_MAX,a,t_train=0;
	size_t nPickSamp=0;
	
	if (stopping.LR_jump>0){
		hTrainData->config.learning_rate *= 2;
	}

	//hTrainData->lossy.InitScore_(hTrainData->config);
	hTrainData->init_score.Init(hTrainData);


	bool isEvalTrain=true;
	DForest curF;
	for (t = 0; t<rounds; t++) {
		skdu.noT = t;
		if (t == 67) {	//仅用于调试
			t = 67;
		}
		FeatVector *hY1 = hTrainData->GetPrecict();
		tpDOWN *hDown = hTrainData->GetDownDirection();
		if(hTrainData->config.T_iterrefine>0)
			IterTrain(t,flag);
		else {			
			if (hEvalData != nullptr) {
				/*if (t > 0) {
					hMTNode hRoot = (dynamic_cast<ManifoldTree*>(forest[t-1]))->hRoot();		//im1 = hRoot->impuri;
					assert(hRoot->nSample() == 0);
				}*/
				err = this->Predict(hEvalData, true, true, true);	//经过校验，同样可以用resumeLast
				//stopping.Add(err,t, isLRjump);
				if (hEvalData->lossy->isOK(0x0, FLT_EPSILON)) {
					eOOB = 0;	printf("\n********* You are so LUCKY!!! *********\n\n");	
				}		
				/*if (isLRjump) {
					double lr0 = hTrainData->config.learning_rate;
					hTrainData->config.learning_rate /= 2;
					printf("\n********* stopping LR(%g=>%g)!!!\t*********\n", lr0, hTrainData->config.learning_rate);
				}*/
				if (hTrainData->feat_salps != nullptr && t>0) {
					hTrainData->feat_salps->SetCost(1-err);
				}
			}
			if (isEvalTrain) {
				err_0 = this->Predict(hTrainData, hEvalData == nullptr, true, true);		//可以继续优化
				if (hTrainData->lossy->isOK(0x0, FLT_EPSILON)) {
					eOOB = 0;	printf("\n********* ERR@Train is ZERO, Break!!! *********\n\n");	return 0x0;
				}
				if (skdu.noT % 500 == 0 && hTrainData->config.verbose==666) {
					a = forest.size() == 0 ? 0 : nzNode*1.0 / forest.size();
					printf("\n\t%d: ERR@%s=%8.5g nNode=%g nPick=[%d,%lld] time=%.5g======\n", skdu.noT, hTrainData->nam.c_str(), err_0,
						a, hTrainData->nPickFeat, nPickSamp, GST_TOC(tick));
				}
				//if (hEvalData == nullptr)
				//	stopping.Add(err_0, t, isLRjump);				
			}
		}		
		if (stopping.isOK(t)) {
			if (hTrainData->config.nMostPrune>0) { 
				this->Prune();		
			}
			/*printf("\n********* early_stopping@[%d,%d]!!! bst=%s ERR@train[%d]=%s overfit=%-8.5g*********\n\n",
				stopping.best_no, stopping.best_round, sLossE.c_str(), skdu.noT, sLossT.c_str(), err - err_0);*/
			if (stopping.isOK(t))
				break;
			else {
				printf("\n********* NO early_stopping!!! continue by Prune!!!\n");
			}
		}
		if (stopping.isOscillate || t==8) {	//仅用于调试
			;// if (hTrainData->config.nMostPrune > 0) { this->Prune(); }
		}
		this->BeforeTrain(hTrainData);
		//gradients = self.loss.negative_gradient(preds, y)
		ManifoldTree *hTree = new ManifoldTree(this, hTrainData, "666_" + to_string(t));
		//if (hEvalData != nullptr)		//case_higgs.py实测确实有BUG
		//	hTree->SetGuideTree(new ManifoldTree(this, hEvalData, "777_" + to_string(t)));
		nPickSamp = hTree->hRoot()->nSample();

		forest.push_back(hTree);
		GST_TIC(t111);
		hTree->Train(flag);				//
		hTree->Adpative_LR(flag);
		/*if (stopping.isOscillate) {		//9/21/2019	无效，难以理解
		//if (forest.size()>50) {
			hTree->DropNodes();
		}*/
		hTree->ClearSampSet();
		nzNode +=hTree->nodes.size();
		maxDepth = max(maxDepth, hTree->maxDepth());
		hTree->ArrTree_quanti = hTree->To_ARR_Tree(hTrainData,true, false);
		hTree->ArrTree_data = hTree->To_ARR_Tree(hTrainData, false,true);
		t_train += GST_TOC(t111);
		
		//TestOOB(hTrainData);
		
		if (t %10==0) {
			//printf("\t%4d: train=%g sec\r\n\n", t+1, GST_TOC(tick));
		}
	}
	string sEval = hEvalData == nullptr ? (isEvalTrain ? hTrainData->nam : "None") : hEvalData->nam;
	string sLossE = hEvalData==nullptr?"":hEvalData->LOSSY_INFO(stopping.e_best), sLossT = hTrainData->LOSSY_INFO(err_0);
	printf("\n====== LOOP=%d: ERR=[~%s,%s] time=%.3g(%.3g) ======\n", skdu.noT, sLossT.c_str(), sLossE.c_str(),GST_TOC(tick), 0);
	double aNode = nzNode*1.0 / forest.size();
	for (i = stopping.best_round ; i<forest.size(); i++) {
		ManifoldTree *hTree = dynamic_cast<ManifoldTree *>(forest[i]);
		//nzNode -= hTree->nodes.size();
		delete forest[i];
	}
	forest.resize(stopping.best_round );
	//if (prune!=nullptr) {		this->Prune();	}

	hTrainData->AfterTrain();
	if (stopping.isOK(t)) {
		printf("\n********* early_stopping@[%d,%d]!!!", stopping.best_no, stopping.best_round);
	}	else {
		printf("\n********* best_@[%d,%d]!!!", stopping.best_no, stopping.best_round);
	}

	printf("\n********* GBRT::Train ERR@train=%s E_best@%s=%s nTree=%d nFeat={%d-%d} aNode=%.6g maxDepth=%d thread=%d" 
		"\n********* train=%g(hTree->Train=%g,tCheckGain=%g,tHisto=%g(%d,%g),tX=%g) sec\r\n", 
		sLossT.c_str(), sEval.c_str(), sLossE.c_str(),forest.size(),stat.nMinFeat,stat.nMaxFeat, aNode, maxDepth, nThread,
		GST_TOC(tick), t_train,FeatsOnFold::stat.tCheckGain, FeatsOnFold::stat.tHisto, HistoGRAM::nAlloc, FeatsOnFold::stat.tSamp2Histo, FeatsOnFold::stat.tX);

	if (nOOB>0)
		TestOOB(hTrainData);
	if (regular == MULTI_TREE) {
		//OnMultiTree(cas, EACH);
		//if (skdu.step<skdu.nStep - 1)	UpdateFeat();
	}
	else {
		//原则上move后需要UpdateFeat,但随着move越来越小，并不需要。1_19.dat/1_22.dat的对比也显示了这一点
		//if (1)	UpdateFeat(0x0);
	}
	//confi.Train( sTitle,cas,flag );
	AfterTrain(hTrainData, 0, nTree);
	//printf("********* GBRT::Train ...... OK\n");
	
	return 0x0;
}

/*
	v0.1	cys
		9/17/2019
	v0.2	cys
		9/29/2019
	没啥效果，郁闷		参见[MORT]_None_0.97700_F8_0.9487_prune.info
*/
int GBRT::Prune(int flag) {
	if (hEvalData == nullptr) {
		printf("********* EnsemblePruning only available when evalue set is not empty!!! ********* ");
		return 0;
	}
	if (prune->nPruneOperation > 0)
		return 0;
	prune->nPruneOperation++;
	int nTree = forest.size(),nPrune = min(hTrainData->config.nMostPrune, nTree /2),i;
	VALID_HANDLE(prune);
	for (i = 0; i < nTree; i++) {
		ManifoldTree *hTree = (ManifoldTree*)forest[i];
		assert(hTree->ArrTree_data != nullptr);			
		if (i < nTree - nPrune)
			continue;
		hPruneData->DeltastepOnTree<double>(*(hTree->ArrTree_data), flag);
		/*if (prune->init_score == nullptr) {
			prune->OnStep(hTree, hPruneData->GetPredict_<double>());
		}
		else*/
		prune->OnStep(hTree, hPruneData->GetDeltaStep());
	}
	double errT, errE = 0;
	if (true || prune->nWeak == prune->nMostWeak) {
		size_t nSamp = nSample();
		int T = nTree /4,i;
		for (i = 0; i < prune->nWeak; i++) {
			prune->cc_0[i] = 1.0;
		}
		GST_TIC(tic)
		prune->Pick(nTree,1,0x0);
		double tPick = GST_TOC(tic);
		//delete prune;		prune = nullptr;	return 0;
		for (i = 0; i < prune->nWeak; i++) {
			double w = prune->cc_1[i];			
			ManifoldTree *hMT=dynamic_cast<ManifoldTree *>(prune->forest[i]);
			hMT->weight= w;
			hMT->ArrTree_quanti->weight= w;
			hMT->ArrTree_data->weight = w;
		}
		prune->Reset4Pick(0x0);
		EARLY_STOPPING stop_old = stopping;
		stopping.Reset();
		DForest trees_0=forest, deads;
		forest.clear();
		hEvalData->GetPrecict()->Empty();
		errE = this->Predict(hEvalData, true, true, true);
		for (i = 0; i < nTree; i++) {
			ManifoldTree *hTree = (ManifoldTree*)trees_0[i];
			if (hTree->weight == 0) {
				deads.push_back(hTree);
				delete hTree;	//deads.push_back(tree);
			}	else {
				forest.push_back(hTree);
				errE = this->Predict(hEvalData, true, true, true);
				if(i<nTree-nPrune)
					assert(stopping.errors[i+1]== stop_old.errors[i + 1]);
			}
		}
		
		errT = this->Predict(hTrainData, hEvalData == nullptr, true, false);
		
		string sLossE = hEvalData == nullptr ? "" : hEvalData->LOSSY_INFO(stopping.e_best), sLossT = hTrainData->LOSSY_INFO(errT);
		printf("\n********* GBRT::Prune[%d] nTree=%d=>%d tPick=%.4g ERR@train=%s err@eval=%s ", prune->nPruneOperation,
			nTree,forest.size(), tPick, sLossT.c_str(), sLossE.c_str()  );/**/
	}
	//delete prune;		prune = nullptr;

	return 0x0;
}

void GBRT::AfterTrain(FeatsOnFold *hData, int cas, int nMulti, int flag) {
	if (model != REGRESSION) {
		float *dtr = hData->distri;
		int i, j, total = hData->nSample(), cls, d1, nCls = hData->nCls, *tag = hData->Tag();
		double mae = 0, rOK = 0;
		for (i = 0; i<total; i++, dtr += nCls) {
			for (cls = -1, d1 = 0, j = 0; j<nCls; j++) {
				if (dtr[j]>d1)
				{
					d1 = dtr[j];	cls = j;
				}
			}
			mae += abs(cls - tag[i]);
			if (cls == tag[i]) {
				rOK += 1;
			}
		}
		mae /= total;			rOK /= total;
		printf("\nmae=%g,rOK=%g\n", mae, rOK);
		return;
	}else{
	}
	double oob0 = 0, err = 0, im1 = 0, a;
	int total = SamplSet.size(), no = 0, nzBag = 0, bag0 = 0;
	if(total==0)
		return;
	eOOB = 0;
	for (auto hCas : SamplSet) {
	//for each(CASE *hCas in SamplSet) {
		if (hCas->nBag>0) {
			hCas->predict /= hCas->nBag;
			a = hCas->label - hCas->predict;
			err += a*a;
		}
	}
	err = sqrt(err / total);
	//	printf( "%d: err=(%5.3g,%5.3g) eOOB=%g nTree=%d nWeak=%g,eta=%g\n",step,im1,hRoot->err,eOOB,forest.size(),nzWeak/(step+1),eta );
	printf("\nerr=(%5.3g)\n", err);
	return;
}

void GBRT::GetYDistri(WeakLearner *hWeak, float *distri, int flag) {
	/*
	double *Y, a;
	int i = 0, nSampe = hTrainData->nSample();
	for each (F4NO *fn in hWeak->samps) {
		int no = fn->pos;			assert(no >= 0 && no<nSampe);
		//feat=TO<float>(hTestData,no);
		Y = hTrainData->Y(no);
		fn->f_1 = a = Y[0];
		if (distri != nullptr)
			distri[i++] = a;
	}*/
}

/*
	似乎可实现简单的特征工程
	v0.1	cys
		7/18/2018
*/
bool GBRT::GetFeatDistri(WeakLearner *hWeak, float *distri, int flag) {
	/*GST_THROW("GBRT::GetFeatDistri is ...");
	BLIT_Diff *hBlit = dynamic_cast<BLIT_Diff*>(hWeak->hBlit);
	GST_VERIFY(hBlit != nullptr, "RF_ShapeRegress::Split is 0");
	float thrsh = hBlit->thrsh, *fu = hTrainData->Feat(hBlit->id), *fv = hTrainData->Feat(hBlit->other);
	int no, i = 0, nTrain = hTrainData->nSample() - nOOB;
	for each (F4NO *fn in hWeak->samps) {
		no = fn->pos;			assert(no >= 0 && no<hTrainData->ldF);
		if (stage == RF_TEST) {
			GST_VERIFY(no >= nTrain, "RF_ShapeRegress::OOB is X");
		}
		else {
			assert(no<nTrain);
		}
		fn->val = fu[no] - fv[no];
		//		fn->val = fu[no];
		if (distri != nullptr)
			distri[i++] = fn->val;
	}*/
	return true;
}

bool GBRT::isPassNode(FeatsOnFold *hData_, hMTNode hNode, int flag) {
	if( hNode->nSample()<Config().min_data_in_leaf * 2 )
		return true;
	if (!hData_->present.isValid(hNode)) {
		assert(hNode->lr_eta==1.0);
		hNode->lr_eta = 0;
		return true;

	}
	int max_dep=Config().max_depth;
	if (max_dep>0 && hNode->depth >= max_dep)
		return true;
	return false;
}


int GBRT::Test(string sTitle, BoostingForest::CASEs& TestSet, int nCls, int flag) {
	GST_TIC(tick);
	stage = RF_TEST;
	int nTest = TestSet.size(), i;
	SamplSet.clear();			SamplSet.resize(nTest);
	copy(TestSet.begin(), TestSet.end(), SamplSet.begin());
	hTestData = nullptr;	// new FeatsOnFold(nullptr, nTest, 0, nCls, 0);
	for (i = 0; i<nTest; i++) {
		//RF_FacePos::POSE_cls *hPose = dynamic_cast<RF_FacePos::POSE_cls*>(SamplSet[i]);
		//if (model == CLASIFY) { hTestData->tag[i] = (int)(hPose->label); }
	}
	nOOB = 0;

	//#pragma omp parallel for num_threads(nThread) private( i ) 
	for (auto hTree : forest) {
	//for each(DecisionTree *hTree in forest) {
		hTree->oob.clear();
		WeakLearner* hRoot = hTree->hRoot();
		BootSample(hTree, hRoot->samps, hTree->oob, hTestData);		assert(hTree->oob.size() == 0);
		hTree->Clasify(hTestData, hRoot->samps, hTestData->distri);
	}
	AfterTrain(hTestData, 0, nTree);
	delete hTestData;			hTestData = nullptr;
	printf("\n********* GBRT nTest=%d,nTree=%d,time=%g*********\n", nTest, nTree, GST_TOC(tick));
	return 0x0;
}

void Representive::dump(int flag) {
	for (auto present : arrPFeat) {
		FeatVector *hFeat = present->hFeat;
		printf("\n\tRepresentive@\"%s\">%.5g", hFeat->nam.c_str(),present->T_min);
	}
}

bool Representive::isValid(const MT_BiSplit *hNode, int flag) {
	for (auto present : arrPFeat) {
		FeatVector *hFeat = present->hFeat;
		size_t nUnique = hFeat->UniqueCount(hNode->samp_set,0x0);
		if (nUnique <= present->T_min) {
			//printf("present=%d<%g\t", nUnique, present->T_min);

			return false;
		}

	}
	return true;
}