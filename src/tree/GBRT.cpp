#include <omp.h>
#include "GBRT.hpp"
#include "../data_fold/Loss.hpp"
#include "../util/Object.hpp"


GBRT::GBRT(FeatsOnFold *hTrain, FeatsOnFold *hEval, double sOOB, MODEL mod_, int nTre_, int flag) : BoostingForest() {
	double rou = 1.0;
#ifdef _DEBUG
#else
#endif
	model = mod_;	
	skdu.cascad = 0;			//skdu.nStep = nStep_;
	int seed = skdu.cascad + 31415927, nTrain = hTrain->nSample(), i, cls;
	skdu.nTree = nTree = rounds = nTre_;

	hTrainData = hTrain;	// new FeatsOnFold(nullptr, nTrain, 0, nCls, 0);
	hEvalData = hEval;
	//	hTrainData=hIn;		hTestData=hTes_;		
	size_t nSamp = hTrainData->nSample( );
	//InitRander(seed);
	
	//lenda = 10;	//MAX(1.0,10-cas*0.5);			
	int nCand = 400;	//有意思
	//eta = 0.1;
	nBlitThread = 8;	//并行导致结果不再可重复
						//nBlitThread=1;	
	maxDepth = hTrain->config.max_depth;
	//int histo = hTrain->config.histo_bins;
	nThread = hTrain->config.num_threads;
	assert(nThread>0 && nThread<32);
	omp_set_num_threads(nThread);
	omp_set_nested(0);
#pragma omp parallel
	nThread = omp_get_num_threads();

	sBalance = 1.0;
	//min_set = MAX(min_set, nSamp / 1000);		hTrain->config.min_data_in_leaf = min_set;
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
	if (nBlitThread>1) {
		for (int i = 0; i<nBlitThread; i++) {
		}
	}
	const char *mod = model==CLASIFY ? "CLASIFY" : "REGRESSION";
	printf("\n\n********* GBRT[%s]\n\tnTrainSamp=%d,nTree=%d,maxDepth=%d regress@LEAF=%s thread=%d feat_quanti=%d...",
		mod,nTrain, nTree, maxDepth, hTrain->config.leaf_regression.c_str(),nThread, hTrain->config.feat_quanti);
	hTrain->config.dump( );
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
double GBRT::Predict(FeatsOnFold *hData_, bool isX,bool checkLossy, bool resumeLast, int flag) {
	GST_TIC(tick);
	//FeatsOnFold *hTrain = GurrentData( );
	size_t nSamp = hData_->nSample(), t;
	hData_->BeforePredict( );
	ManifoldTree *lastTree = nullptr;
	bool isResetZero=false;		//init_score带来了很多问题，需要重新设计,参见ManifoldTree::Train之AddScore
	if (resumeLast && forest.size()>0) {
		lastTree = (ManifoldTree*)forest[forest.size() - 1];
	}else{
		hData_->GetPrecict()->Empty();
		if (forest.size() == 0) {
			hData_->GetPrecict()->Set(hData_->init_score.step);
			isResetZero=true;
		}
	}

	FeatVec_T<tpDOWN>* predict = dynamic_cast<FeatVec_T<tpDOWN>*>(hData_->GetPrecict());
	assert(predict!=nullptr);
	tpDOWN *allx = predict->arr();

	for (auto tree : forest) {
	//for each(ManifoldTree*hTree in forest) {
		ManifoldTree *hTree = dynamic_cast<ManifoldTree *>(tree);
		assert(hTree!=nullptr);
		if (lastTree != nullptr && hTree != lastTree)
			continue;
		bool isNodeMajor = true;
		if (true) {		//data-major	似乎快一些
			ARR_TREE arrTree;
			if (hTree->To_ARR_Tree(hData_,arrTree)) {
				if(hData_->isQuanti)
					isNodeMajor = !hData_->PredictOnTree<tpQUANTI,double>(arrTree,flag);
				else
					isNodeMajor = !hData_->PredictOnTree<float, double>(arrTree, flag);
			}
		}
		if(isNodeMajor) {			//node-major
			hMTNode root = hTree->hRoot();
			bool isSplit =  root->nSample() == 0;
			if (hData_->hMove != nullptr)
				;// hData_->hMove->BeforeStep(hData_->samp_set, allx, 0x0);
			if (isSplit) {
				//root->samp_set = hData_->samp_set;				root->samp_set.isRef = true;
				root->samp_set.SampleFrom(hData_,nullptr, hData_->nSample(),-1);
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
		if (hData_->hMove != nullptr)
			;// hData_->hMove->AfterStep(hData_->samp_set, allx);
	}
	if(checkLossy)
		hData_->lossy->Update(hData_,this->skdu.noT,0x0);

	/*double nNode=0;
	for each(ManifoldTree*hTree in forest) {
		nNode+=hTree->nodes.size();
		hTree->ClearSampSet( );		
	}
	nNode= forest.size()==0 ? nNode : nNode*1.0/ forest.size();*/
	double err=DBL_MAX, y2 = DBL_MAX;
	if (checkLossy) {
		//on the objective
		if (hData_->config.objective == "binary") {
		}
		else {
		}
		//on the eval_metric
		if (hData_->config.eval_metric == "mse") {
			err = hData_->lossy->err_rmse;
			err = err*err;
		}	else if (hData_->config.eval_metric == "mae") {
			err = hData_->lossy->err_mae;
		}	else if (hData_->config.eval_metric == "logloss") {
			err = hData_->lossy->err_logloss;
		}	else if (hData_->config.eval_metric == "auc") {
			err = 1-hData_->lossy->err_auc;
		}

		if (BIT_TEST(hData_->dType, FeatsOnFold::DF_EVAL)) {
			if ((skdu.noT < 100 && skdu.noT % 5 == 0) || skdu.noT % 500 == 0) {
				printf("%s_%d=%-8.5g ", hData_->nam.c_str(),skdu.noT,  err);
				if(skdu.noT > 100)	printf("tX=%.3g ", FeatsOnFold::stat.tX);
			}

		}
		
	}
	if (isResetZero) {
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

void EARLY_STOPPING::Add(double err, int flag) {
	errors.push_back(err);
	if (err < e_best) {
		e_best=err;		best_no=errors.size()-1;
	}

}
bool EARLY_STOPPING::isOK() {
	if( errors.size()<early_round )
		return false;
	double e_last = errors[errors.size()-1];
	if (best_no<= errors.size()- early_round) {
		assert(e_last>=e_best);
		return true;
	}
	return false;
}

int GBRT::Train(string sTitle, int x, int flag) {
	GST_TIC(tick);
	eOOB = 1.0;
	size_t nSamp = nSample(), t;
	assert(forest.size() == 0);
	mSum = 0;	// mSum.setZero();	

	//nzWeak=0.0;
	//参见BoostingForest::Train();
	stage = RF_TRAIN;
	int nIns = 0, no = 0, total, i, j, nzNode=0;
	//FeatsOnFold *hData=curF[0]->hData;
	total = hTrainData->nSample();			
	stopping.early_round = hTrainData->config.early_stopping_round;
	float *distri = hTrainData->distri, *dtr = nullptr, tag, d1, rOK = 0;
	double err_0= DBL_MAX,err=DBL_MAX,a;
	size_t nPickSamp=0;
	
	//hTrainData->lossy.InitScore_(hTrainData->config);
	hTrainData->init_score.Init(hTrainData);


	bool isEvalTrain=true;
	DForest curF;
	for (t = 0; t<rounds; t++) {
		skdu.noT = t;
		if (isEvalTrain) {
			err_0 = this->Predict(hTrainData,false,true, true);		//可以继续优化
			if (hTrainData->lossy->isOK(0x0,FLT_EPSILON)) {
				eOOB = 0;	printf("\n********* ERR@Train is ZERO, Break!!! *********\n\n");	return 0x0;
			}
			if (skdu.noT % 500 == 0) {
				a = forest.size() == 0 ? 0 : nzNode*1.0 / forest.size();
				printf("\n====== %d: ERR@%s=%8.5g nNode=%g nPickFeat=%d nPickSamp=%lld time=%.3g======\n", skdu.noT, hTrainData->nam.c_str(), err_0, 
					a, hTrainData->nPickFeat, nPickSamp, GST_TOC(tick));
			}
			if (hEvalData == nullptr)
				stopping.Add(err_0);
		}
		if (hEvalData != nullptr) {
			if (t > 0) {
				hMTNode hRoot = (dynamic_cast<ManifoldTree*>(forest[t-1]))->hRoot();		//im1 = hRoot->impuri;
				assert(hRoot->nSample() == 0);
			}
			err = this->Predict(hEvalData, true, true, true);	//经过校验，同样可以用resumeLast
			stopping.Add(err);
			if (hEvalData->lossy->isOK(0x0, FLT_EPSILON)) {
				eOOB = 0;	printf("\n********* You are so LUCKY!!! *********\n\n");	return 0x0;
			}			
		}	
		if (stopping.isOK()) {
			printf("\n********* early_stopping@[%d]!!! bst=%g ERR@train[%d]=%-8.5g overfit=%-8.5g*********\n\n",
				stopping.best_no, stopping.e_best, skdu.noT, err_0, err - err_0);
			break;
		}

		this->BeforeTrain(hTrainData);
		//gradients = self.loss.negative_gradient(preds, y)
		ManifoldTree *hTree = new ManifoldTree(this, to_string(666) + "_" + to_string(t));
		nPickSamp = hTree->hRoot()->nSample();
		forest.push_back(hTree);
		hTree->Train(flag);				//
		
		nzNode +=hTree->nodes.size();
		//TestOOB(hTrainData);
		
		if (t %10==0) {
			//printf("\t%4d: train=%g sec\r\n\n", t+1, GST_TOC(tick));
		}
	}
	//printf("\n====== %d: ERR@%s=%8.5g time=%.3g(%.3g) ======\n", skdu.noT, hTrainData->nam.c_str(), err_0,GST_TOC(tick), 0);
	for (i = stopping.best_no + 1; i<forest.size(); i++) {
		delete forest[i];
	}
	forest.resize(stopping.best_no + 1);
	hTrainData->AfterTrain();
	string sEval = hEvalData == nullptr ? (isEvalTrain ? hTrainData->nam : "None") : hEvalData->nam;
	printf("\n********* GBRT::Train nTree=%d aNode=%.6g ERR@train=%-8.5g err@%s=%.8g thread=%d train=%g(tX=%g) sec\r\n", 
		forest.size(), nzNode*1.0/forest.size(), err_0, sEval.c_str(), stopping.e_best,nThread, GST_TOC(tick), FeatsOnFold::stat.tX);

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

bool GBRT::isPass(hMTNode hNode, int flag) {
	if( hNode->nSample()<Config().min_data_in_leaf * 2 )
		return true;
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