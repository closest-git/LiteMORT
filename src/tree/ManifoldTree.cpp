#include "ManifoldTree.hpp"
#include "BoostingForest.hpp"
#include "BiSplit.hpp"
#include <random>
#include <thread>
#include <time.h>
#include <thread>
#include <stack>
#include "../data_fold/Loss.hpp"
#include "../util/Object.hpp"
#include "../EDA/SA_salp.hpp"
#include "BiSplit.hpp"

//double WeakLearner::minSet=0.01;
//double WeakLearner::minGain=0.01;
static	char ssPath[256];		//only for debug

#if defined (_MSC_VER)  // Visual studio
    #define thread_local __declspec( thread )
#elif defined (__GCC__) // GCC
    #define thread_local __thread
#endif


double Impurity( int nCls,int nz,int *distri ){
	double e=0.0,p;
	int i,n1=0;
	for( i=0; i<nCls; i++ ){
		if( distri[i]>0 )		{																								
			p=distri[i]*1.0/nz;		
			e+=p*(1-p);	//gini impurity
			//e-=p*log(p);	///Shannon entropy
			n1+=distri[i];	
		}
	}
	if( n1!=nz )	GST_THROW("Impurity:n1!=nz" );
	return e;
}

WeakLearner::WeakLearner( DecisionTree *hTr,arrPFNO&sam_,int d,int flag ) :
		samps(sam_),feat(-1),thrsh(0.0),left(nullptr),right(nullptr),hTree(hTr),distri(nullptr),hBlit(nullptr),depth(d){
	int i,no,cls;
	bool isRoot=samps.size()==0;

	FeatsOnFold *hDat=hTree->GetDat( );
	int *tag = hDat->Tag();
	if( samps.size()==0 ){		//
		int nSample=hDat->nSample();
		for( i=0; i<nSample; i++ )	{
			int cls=tag[i];
			F4NO *hfn=new FLOA_NO<float>(-1,i,cls);
			samps.push_back( hfn );
		}
	}
	int nSamp=samps.size( ),*distri=new int[hDat->nCls]();
	for ( arrPFNO::iterator it=samps.begin(); it!=samps.end();it++ )	{
		F4NO *hfn=*it;
		no = hfn->pos;		cls=tag[no];
		assert( cls>=0 && cls<hDat->nCls );
		distri[cls]++;
	}
	impuri=Impurity( hDat->nCls,nSamp,distri);
	delete[] distri;
}

double WeakLearner::s_plit( int cur,int *nEachCls,int *nos,double &thrsh,int flag ){
/*	FeatsOnFold *hData=hTree->GetDat( );
	int i,j,cls,nCls=hData->nCls,nSamp=samps.size(),nLeft=0,nRight=nSamp,no,ldX=hData->ldX;
	int h_0,h_1,*ptr=nullptr,*ptr1,dim,pos,iv,*permut=nos+nSamp;
	double gain=0.0,g,g0=impuri,a0,a1,a;
	const float *feat=hData->Feat(cur);//,*dat=TO<float>(hData)+cur;	
	a1=-DBL_MAX,	a0=DBL_MAX;
	memset( nEachCls,0x0,sizeof(int)*nCls );
	//for each ( F4NO *hfn in samps )	{
	for( j=0; j<nSamp;j++ )	{
		no = nos[j];			cls=hData->tag[no];
		a=feat[no];			//hfn->val=a=dat[no*ldX];			
		a1=MAX(a1,a);			a0=MIN(a0,a);
		nEachCls[cls]++;
	}
	if( a1==a0 )
		return -DBL_MAX;
	h_0=(int)(floor(a0));		h_1=(int)(ceil(a1));
	if( h_0==h_1 )
		return -DBL_MAX;
	if( !hData->qOrder ){	//无法使用下面的快速排序
		for( j=0; j<nSamp;j++ )	{
			no = samps[j]->pos;		samps[j]->val=feat[no];
		}
		std::sort( samps.begin(),samps.end(),F4NO::isPSmall );
		for( j=0; j<nSamp;j++ )	{
			permut[j]=samps[j]->pos;
		}
	}else{
		dim=h_1-h_0+1;		ptr=new int[(dim+1)*2](),	ptr1=ptr+dim+1;		
//		for each ( F4NO *hfn in samps )	{
		for( j=0; j<nSamp;j++ )	{
			no = nos[j];			iv=(int)(feat[no])-h_0;		
			assert(iv<dim && iv>=0);
			ptr[iv]++;		
		}
		ptr[dim]=nSamp;
		for( j=dim-1; j>=0; j-- )	
		{	ptr[j]=ptr[j+1]-ptr[j];		ptr1[j]=ptr[j];	}
		assert( ptr[0]==0 );

		for( j=0; j<nSamp;j++ )	{
			no = nos[j];			iv=(int)(feat[no])-h_0;		
			permut[ptr1[iv]++]=no;
		}
		for( j=0; j<dim; j++ )	{	assert(ptr[j+1]==ptr1[j]);	}
	//	for( j=0;j<nSamp;j++ )		{	samps[j]=arrFN[j];	}
	//	delete[] arrFN;		
		if( ptr!=nullptr ) delete[] ptr;
	}
		
	int *dL=nEachCls+nCls,*dR=nEachCls;
	for( j=0;j<nCls;j++ )	dL[j]=0;
	
	//F4NO *hfn=*(samps.begin()+minSet);
	if( 1 ){
//		for each( hfn in samps ){
		for( j=0; j<nSamp;j++ )	{
			no = permut[j];				
			cls=hData->tag[no];		
			if( nLeft<minSet )	
			{	goto LOOP;	}
			if( nRight<minSet )
				break;
			if( feat[no]==a )
			{	goto LOOP;	}
			g = g0-(nLeft*Impurity( nCls,nLeft,dL)+nRight*Impurity( nCls,nRight,dR))/nSamp;
			if( g>gain )
			{	gain=g;		thrsh=(feat[no]+a)/2;	}
		LOOP:
			assert( j==0||feat[no]>=a );		a = feat[no];
			dL[cls]++;		nLeft++;	dR[cls]--;	nRight--;			
		}
	}else{
		arrPFNO::iterator iend=samps.end()-minSet;
		for ( arrPFNO::iterator it=samps.begin()+minSet; it!=iend;it++ )	{
			F4NO *hfn=*it;
			if( hfn->val==(*(it+1))->val )
				continue;
			cls=hData->tag[hfn->pos];			
			dL[cls]++;		nLeft++;	dR[cls]--;	nRight--;
			g = g0-(nLeft*Impurity( nCls,nLeft,dL)+nRight*Impurity( nCls,nRight,dR))/nSamp;
			if( g>gain )
			{	gain=g;		thrsh=(hfn->val+(*(it+1))->val)/2;	}
		}	
	}
	
	return gain;*/
	return 0;
}

void split_samps( float *feat,float thrsh,arrPFNO &samps,arrPFNO &fnL,arrPFNO&fnR ){
	int no;
	for ( arrPFNO::iterator it=samps.begin(); it!=samps.end();it++ )	{
		F4NO *hfn=*it;			no = hfn->pos;
	//	assert( no==nos[i++] );
		hfn->val=feat[no];	//dat[no*ldX];
		if( hfn->val<thrsh ){
			fnL.push_back(*it);
		}else
			fnR.push_back(*it);
	}	
	samps.clear( );	
}

bool WeakLearner::Split( int flag ){
	return true;
}




DecisionTree::DecisionTree( BoostingForest *hF,FeatsOnFold *hD,int flag ):hForest(hF), hData_(hD),root(nullptr)	{
	arrPFNO boot;
	assert( hData_->nSample()>0 );
	//BootSample( boot,oob,hData );
	hF->BootSample( this,boot,oob, hData_);
	//hTrainData = hD;		
	root=nullptr;
	switch( hF->model ){
	case BoostingForest::REGRESSION:
		root=new RandRegress( this,boot,0 );		
		break;
	default:
		root=new RandClasify( this,boot );
		break;
	}
	//WeakLearner::minGain=MIN(WeakLearner::minGain,root->impuri*0.001);
	//WeakLearner::minGain=MIN(WeakLearner::minGain,root->impuri*0.00001);
	GST_VERIFY(root!=nullptr,"DecisionTree has no root!" );

	ginii.clear( );
	int i,nFeat=hD->nFeat();
	for( i=0; i<nFeat; i++ )
		ginii.push_back( F4NO(0.0,i) );
}
DecisionTree::~DecisionTree( )	{
	WeakLearners nodes;
	GetNodes( nodes );
	for (auto hWL : nodes) {
	//for each( WeakLearner* hWL in nodes ){
		assert(hWL->samps.size()==0);		delete hWL;
	}
	nodes.clear( );
	for (auto hfn : oob)
	//for each( F4NO *hfn in oob )
	{	delete hfn;			}
	oob.clear( );
}

void DecisionTree::GetLeaf(  WeakLearners&vLeaf,int flag ){
	vLeaf.clear();
	stack<WeakLearner*> stk;
	stk.push( hRoot() );	
	while( !stk.empty() ){
		WeakLearner *hWL=stk.top();		stk.pop();
		if( hWL->left!=nullptr )	{
			stk.push(hWL->right);		stk.push(hWL->left);		
		}else
			vLeaf.push_back( hWL );
	}
	assert( stk.empty() );
}
void DecisionTree::GetNodes(  WeakLearners&vNodes,int flag ){
	vNodes.clear();
	if(hRoot()==nullptr )
		return;
	stack<WeakLearner*> stk;
	stk.push( hRoot() );	
	while( !stk.empty() ){
		WeakLearner *hWL=stk.top();		stk.pop();
		vNodes.push_back( hWL );
		if( hWL->left!=nullptr )	{
			stk.push(hWL->right);		stk.push(hWL->left);		
		}
	}	
	assert( stk.empty() );
}

void DecisionTree::Train(  int flag ){
	//if( hForest->skdu.noT%10==0 )
	//	printf( "\n********* Tree %d......\n",hForest->skdu.noT );
	WeakLearner* root=hRoot( );
	int nIns=0,nz=0,i;
	impurity=0;
	double a;
	WeakLearners vLeaf;
	do{
		GetLeaf( vLeaf );
		nIns=0;				
		//printf( "\rTree_%s(%d,%p^%p)...stage=%d...Split...",name,vLeaf.size(),root->left,root->right,hForest->stage );
		for (auto hWL : vLeaf) {
		//for each( WeakLearner* hWL in vLeaf ){
			if( hWL->Split( ) )
				nIns+=2;		
		}
	}while( nIns>0 );
	if(0){
		std::sort( ginii.begin(),ginii.end(),F4NO::isBig );
		for( i=0;i<20;i++ ){
			printf( "%d-%g ",ginii[i].pos,ginii[i].val );
		}
	}

	GetLeaf( vLeaf );
	a=0;		hForest->skdu.noLeaf=0;
	for (auto hWL : vLeaf) {
	//for each( WeakLearner* hWL in vLeaf ){
		a+=hWL->impuri;		nz+=hWL->samps.size();
		hWL->AtLeaf( );
		if( !hForest->isDumpLeaf )
		{	hWL->nLastSamp=hWL->samps.size();		hWL->ClearSamps( );	}
		hForest->skdu.noLeaf++;
	}
	assert( nz+oob.size()<=hData_->nSample() );
	nLeaf = vLeaf.size( );
	impurity = a;
	if( nLeaf==1 )
	{	printf( "\n>>>>>>RF_%s failed to split!!!nIns=%d nz=%d impurity=%g",name.c_str(),nIns,nz,impurity );	}

	//printf( "\n%d...OK",hForest->skdu.noT );
}


ManifoldTree::ManifoldTree(BoostingForest *hF, FeatsOnFold *hData, string nam_, int flag)	{
	hForest = hF, name = nam_;
	hData_ = hData;
	//FeatsOnFold *hData = hF->GurrentData();
	int rnd_seed=hF->skdu.noT;
	//hData->samp_set.Shuffle(rnd_seed);
	MT_BiSplit *root = new MT_BiSplit(hData, 0, rnd_seed);
	root->id = nodes.size();		 nodes.push_back(root);
	//leafs.push_back(root);
	//leafs.push(root);

	size_t i, nSamp = hData->nSample();
	/*samp_folds.resize(nSamp);
	for (i = 0; i < nSamp; i++) {
		samp_folds[i] = 0;
	}*/
}


void ManifoldTree::AddNewLeaf(hMTNode hNode, FeatsOnFold *hData_, const vector<int> &pick_feats, int flag) {
	hNode->gain_ = 0;			
	if (hForest->isPass(hNode)) {
		;
	}
	else {		
		std::string leaf_optimal = hData_->config.leaf_optimal;
		assert(hNode->feat_id == -1 && hNode->nSample()>=hData_->config.min_data_in_leaf * 2);
		//if( hNode->impuri>0)		//optimal == "lambda_0"时,impuri不对
			hNode->CheckGain(hData_,pick_feats, 0);
			if (hGuideTree != nullptr) {
				FeatsOnFold *hGuideData = hGuideTree->hData_;
				hMTNode hBlit = hGuideTree->nodes[hNode->id];
				hBlit->Observation_AtLocalSamp(hGuideData);
				hBlit->feat_id = hNode->feat_id;	 hBlit->fruit = hNode->fruit;
				hGuideTree->GrowLeaf(hBlit, "guide_gain", false);
				//hGuide->lossy->Update(hData_, 0, 0x0);
				//double a = 1 - hGuide->lossy->err_auc;
				hBlit->left->Observation_AtLocalSamp(hGuideData);		hBlit->right->Observation_AtLocalSamp(hGuideData);
				double imp = hBlit->right->impuri + hBlit->left->impuri;
				double gain = FLOAT_ZERO(imp - hBlit->impuri,imp) ;
				hNode->gain_ = gain;			assert(gain >= 0);
				hGuideTree->DelChild(hBlit);
				hBlit->fruit = nullptr;/**/
			}
	}
	leafs.push(hNode);

}

void ManifoldTree::DelChild(hMTNode hNode, int flag) {
	vector<hMTNode> cands;
	if (hNode->left != nullptr) {
		cands.push_back(hNode->left);		hNode->left = nullptr;
	}
	if (hNode->right != nullptr) {
		cands.push_back(hNode->right);		hNode->right = nullptr;
	}
	for (auto hNode : cands) {
		nodes.erase(std::remove(nodes.begin(), nodes.end(), hNode), nodes.end());
		delete hNode;
	}
	cands.clear();
}
/*
	v0.1	cys
		8/2/2018
*/
void ManifoldTree::GrowLeaf(hMTNode hBlit, const char*info, bool isAtLeaf, int flag) {
	//assert(hData_->atTrainTask());
	assert(hBlit->isLeaf());
	size_t nSamp= hBlit->nSample();
	//MT_BiSplit *hBlit = dynamic_cast<MT_BiSplit *>(node);		assert(hBlit != nullptr);
	hBlit->left = new MT_BiSplit(hBlit),			hBlit->right = new MT_BiSplit(hBlit);
	hBlit->left->brother = hBlit->right;			hBlit->right->brother = hBlit->left;

	//leafs.erase(std::remove(leafs.begin(), leafs.end(), hBlit), leafs.end());
	hBlit->left->id = nodes.size( );		
	nodes.push_back(hBlit->left);			//leafs.push_back(hBlit->left);
	hBlit->right->id = nodes.size();		
	nodes.push_back(hBlit->right);			//leafs.push_back(hBlit->right);
	
	//FeatVector *Feat = hData->Feat(hBlit->feat_id);
	GST_TIC(t1);
	hData_->SplitOn(hBlit);
	//FeatsOnFold::stat.tX += GST_TOC(t1);
	if (isAtLeaf) {
		hData_->AtLeaf(hBlit->left);
		hData_->AtLeaf(hBlit->right);
	}

	//FeatsOnFold::stat.tX += GST_TOC(t1);
	size_t nLeft= hBlit->left->nSample(), nRight = hBlit->right->nSample();
	double imp = hBlit->right->impuri+ hBlit->left->impuri;
	//node->Dump(info, 0x0);
	double thrsh = 1.0e-5;
	std::string leaf_optimal = hData_->config.leaf_optimal;
	if(hBlit->gain_train>thrsh)	{//impuri match校验。当gain太小，可忽略
		bool isMatch = false;
		if (leaf_optimal == "grad_variance") {
				isMatch = fabs(hBlit->impuri-imp- hBlit->gain_train)<= imp*thrsh;
			assert( imp<= hBlit->impuri);
		}	else if(leaf_optimal == "lambda_0") {	//"taylor_2";	
			isMatch = fabs(hBlit->impuri - imp + hBlit->gain_train) <= imp*thrsh;
			assert(imp >= hBlit->impuri);
		}
		if(!isMatch){
			string sX = hBlit->fruit->sX;
			printf("\t!!!<%d:dad-child=%g gain=%g>!!!\t\n%s\t%s\t%s", hBlit->id,hBlit->impuri - imp, hBlit->gain_train,sX.c_str(),
				hBlit->left->sX.c_str(), hBlit->right->sX.c_str());
			//throw "ManifoldTree::Grow Leaf !!!isMatch!!!";
		}
	}

}

/*
	参见MT_BiSplit::MT_BiSplit
	1-25测试，没有效果!!!（似乎batch的样本变小，其二分类的泛化效果就不好）

*/
void ManifoldTree::BeforeEachBatch(size_t nMost, size_t rng_seed, int flag) {
	//this->ClearSampSet();
	int nTree = hForest->forest.size(), iter = 0;
	//FeatsOnFold *hData_ = hForest->GurrentData();
	hMTNode root = hRoot();
	size_t nLeft = root->samp_set.nLeft, nRigt = root->samp_set.nRigt,nSamp= root->samp_set.nSamp;
	root->samp_set.SampleFrom(hData_,nullptr, nMost, rng_seed, 0);

	for (auto node : nodes) {
	//for each(hMTNode node in nodes) {
		SAMP_SET &set = node->samp_set;
		if( node->isLeaf() ){	//和FeatsOnFold:;AtLeaf一致
			node->Observation_AtLocalSamp(hData_);
			node->feat_id=-1;			node->gain_train = 0;
			if (node->fruit != nullptr) {
				delete node->fruit;		 node->fruit = nullptr;
			}
		}	else {					//参见ClearSampSet();
			//set.nLeft = 0;		set.nRigt = 0;
			node->left->samp_set.ClearInfo();
			node->right->samp_set.ClearInfo();
					
			hData_->SplitOn(node);
		}
	}/**/
}

/*
	注意：不管X,Y怎么变化，ManifoldTree的目标函数是negative_gradient!!!	
	其数据类型始终是tpDOWN

	v0.2	cys	
		4/25/2019
*/
void ManifoldTree::Train(int flag) {	
	int nTree=hForest->forest.size( ),iter=0;
	//FeatsOnFold *hData_ = hForest->GurrentData();
	bool isBatch = hData_->config.batch < 0.9;
	tpDOWN *yDown = hData_->GetDownDirection();
	char info[2000];
	size_t pick,i, nSamp = hData_->nSample(),nPass=0, nz=0;

	hMTNode hBest=nullptr,root = this->hRoot();
	double imp_0 = root->impuri, gain = 0,a;
	//printf("\n-----TREE(nodes=%d,leaf=%d),imp_0=%g,", nodes.size(), leafs.size(), imp_0 );
	vector<int> pick_feats;
	hData_->nPick4Split(pick_feats, hData_->rander_feat, hForest,-1 );
	if(true){
		for (auto no : pick_feats) { 
			hData_->feats[no]->PerturbeHisto(hData_);
		}
	}
	if (this->hForest->skdu.noT % 500 == 0) {
		tpSAMP_ID *samps = root->samp_set.samps;
		size_t nSamp = root->samp_set.nSamp;
		printf("\t\tsamps={%d,%d,...,%d}",samps[0], samps[nSamp / 2], samps[nSamp - 1]);
		if (pick_feats.size() > 1) {
			printf("\tpick_feats={%d...%d...%d}\t",
				pick_feats[0], pick_feats[pick_feats.size()/2], pick_feats[pick_feats.size()-1]);
		}
	}
	
	AddNewLeaf(root, hData_, pick_feats);
	gain = 0;
	while (true) {		//参见GBRT::GetBlit
		if(leafs.size()>=hData_->config.num_leaves )
			break;
		gain = 0;		hBest=nullptr;		nz = 0;
		if (isBatch && iter%10==0) {
			//Update root sample and update Functional at each nodes
			BeforeEachBatch(hData_->config.batch*nSamp,42+ leafs.size());
		}
		hMTNode hBest = leafs.top();
		if (hBest->id == 2)
			hBest->id = 2;		//仅用于调试
		if (hBest->gain_train > 0)		{
			gain = hBest->gain_train;			leafs.pop();
		}		else {
			hBest = nullptr;
		}

		
		iter++;
		MT_Nodes new_leafs;
		if (hBest != nullptr) {
			sprintf(info, "Grow@Leaf_%d gain=%8g", hBest->id, hBest->gain_train);
			GrowLeaf(hBest,info,true);
			if (hGuideTree != nullptr) {
				hMTNode hSplit = hGuideTree->nodes[hBest->id];
				hSplit->feat_id = hBest->feat_id;	 hSplit->fruit = hBest->fruit;
				hGuideTree->GrowLeaf(hSplit, info,false);
				hSplit->fruit = nullptr;/**/
			}
			//hBest->Dump(info, 0x0);
			assert( hBest->left->nSample()==hBest->fruit->nLeft && hBest->right->nSample()==hBest->fruit->nRight);
			if (hBest->left->nSample() < hBest->right->nSample()) {
				new_leafs.push_back(hBest->left);		new_leafs.push_back(hBest->right);
			}	else {
				new_leafs.push_back(hBest->right);		new_leafs.push_back(hBest->left);
			}			
		}	else {
			break;
		}
		//GST_TIC(t1);
		//FeatsOnFold::stat.tX += GST_TOC(t1);
		for (auto leaf : new_leafs)		{
			AddNewLeaf(leaf, hData_, pick_feats);
		}
		//if(hBest->left!=nullptr)
		//	AddNewLeaf(hBest->left,hData_, pick_feats);
		//if (hBest->right != nullptr)
		//	AddNewLeaf(hBest->right, hData_, pick_feats);
	}

	//GetLeaf(vLeaf);
	a = 0;		nz = 0;
	hForest->skdu.noLeaf = 0;
	for (auto node : nodes) {
	//for each(hMTNode node in nodes) {
		if (node->isLeaf()) {
			//node->Observation_AtLocalSamp(hData_);
			nLeaf++;
			size_t nSamp= node->nSample();
			a += node->impuri;		nz += node->nSample();
			hForest->skdu.noLeaf++;
			if (hData_->config.leaf_regression=="linear" && nSamp>2 ) {
				assert( node->feat_regress != -1);	//node->feat_id==-1 &&
				FeatVector *hF_ = hData_->Feat(node->feat_regress);
				node->regression= hF_->InitRegression(hData_, node,0x0);
			}
			else if(hData_->config.leaf_regression == "histo_mean" && nSamp>2) {
				//注意 在Grow Leaf时，每个节点的impuri,fruit已被初始化
				if( node->impuri>0 )
					node->CheckGain(hData_, pick_feats, 0);
			}
		}
	}
	assert(nz + oob.size() <= hData_->nSample());
	gain = imp_0-a;
	if (nLeaf == 1)	{
		printf("\n>>>>>>ManifoldTree_%s failed to split!!! nz=%d gain=%g(%g->%g)", name.c_str(),  nz, gain,imp_0,a);
	}	else {
		if (nTree == 1) {		//不是很合理
			AddScore(&(hData_->init_score));
		}
	}
		;//printf("    ---- node=%d gain=%g(%g->%g)\n", nodes.size(), gain, imp_0, a);
	//Dump( );

	ClearSampSet( );		//优化，还需predict
	for (auto node : nodes) {
		for (auto pa : node->H_HISTO) {
			HistoGRAM* histo = pa;
			if(histo!=nullptr)
				delete histo;
		}
		node->H_HISTO.clear();
	}

	//printf( "\n%d...OK",hForest->skdu.noT );
}

/*
	v0.1	cys
*/
void ManifoldTree::ClearSampSet() {
	for (auto node : nodes) {
	//for each(hMTNode node in nodes) {		
		node->samp_set.ClearInfo();
		//node->lef
	}
}

/*
	v0.1	cys
*/
bool ManifoldTree::To_ARR_Tree(FeatsOnFold *hData_, ARR_TREE &arrTree, int flag) {
	bool isQuanti = hData_->isQuanti;		//predict,test对应的数据集并没有格子化!!!
	int nNode = nodes.size(),no=0;
	arrTree.Init(nNode);
	no = 0;
	for (auto node : nodes) {
		node->id = no++;
	}
	no = 0;
	for (auto node : nodes) {
		arrTree.feat_ids[no] = -1;
		arrTree.left[no] = -1;		arrTree.rigt[no] = -1;
		if (node->left != nullptr) {	//参见void SplitOn(...
			if (node->fruit->isY)
				return false;
			arrTree.feat_ids[no] = node->feat_id;
			arrTree.left[no] = node->left->id;
			assert(node->right != nullptr);
			arrTree.rigt[no] = node->right->id;
			//arrTree.thrsh_step[no] = isQuanti ? node->fruit->T_quanti : node->fruit->thrshold;
			arrTree.thrsh_step[no] = node->fruit->Thrshold(isQuanti);
		}	else {
			arrTree.thrsh_step[no] = node->GetDownStep();
		}
		no = no + 1;
	}
	return true;
}

void ManifoldTree::Dump( int flag ){
	//FeatsOnFold *hData_ = hForest->GurrentData();
	if(hData_->config.verbose<=0)
		return;
	bool isDistri = false;
	printf("\n------Tree=%d num_leaves=%d------", hForest->forest.size()-1,leafs.size());
	printf("\n\tsplit_feature=");
	for (auto node : nodes) {
	//for each(hMTNode node in nodes) {
		if (node->isLeaf()) continue;
		FeatVector *hFeat = hData_->Feat(node->feat_id);
		size_t nSamp = node->nSample();
		//printf("%d ", node->feat_id);
		printf("%s ", hFeat->nam.c_str());
		if (isDistri) {
			
			FeatVec_Q *hFQ = dynamic_cast<FeatVec_Q*>(hFeat);
			bool isQuanti = hFQ != nullptr;
			
			Distribution disX, disY,disDown;
			
			hFeat->Observation_AtSamp(hData_->config, node->samp_set, disX);
			if (hFQ != nullptr) {
			}
			else {
			}
			hData_->GetY()->Observation_AtSamp(hData_->config, node->samp_set, disY);		
			//hData_->lossy->down.Observation_AtSamp(hData_->config, node->samp_set, disY);
			//hFeat->Samp2Histo(hData_, node->samp_set, disX.histo, hData_->config.feat_quanti);
			printf("\n\tdisX@");			disX.Dump(node->feat_id, false,flag);
			printf("\tdisY@");				disY.Dump(node->feat_id,false,flag);
			//printf("\tdisDown@");				disY.Dump(node->feat_id, false, flag);
		}
	}
	printf("\n\tsplit_gain=");
	for (auto node : nodes) {
	//for each(hMTNode node in nodes) {
		if (node->isLeaf()) continue;
		printf("%.7g ", node->gain_train);
	}
	printf("\n\tthreshold=");
	for (auto node : nodes) {
	//for each(hMTNode node in nodes) {
		if (node->isLeaf()) continue;
		printf("%.8g ", node->fruit->Thrshold(false));		
	}
	printf("\n\tleaf_value=");
	for (auto node : nodes) {
	//for each(hMTNode node in nodes) {
		if (node->isLeaf()) {
			printf("%.7g ",node->GetDownStep() );
		}
	}
	printf("\n\tleaf_count=");
	for (auto node : nodes) {
	//for each(hMTNode node in nodes) {
		if (node->isLeaf()) {
			printf("%d ",node->samp_set.nSamp);
		}
	}
	printf( "\n\n",hForest->skdu.noT );

}

void ManifoldTree::AddScore(INIT_SCORE *score, int flag) {
	double bias=score->step;
	assert(!IS_NAN_INF(bias));
	if(bias ==0)
		return;
	for (auto node : nodes) {
	//for each(hMTNode node in nodes) {
		if (node->isLeaf()) {
			node->down_step += bias;
		}
	}
}

//调用之前distri必须清零
void DecisionTree::Clasify( FeatsOnFold *hSamp,arrPFNO &points,float *distri,int flag ){
	int nSamp=points.size( ),sum=0,nz,no,nClass=hSamp->nCls,j;
	float *model,*dtr=nullptr;
	WeakLearners vLeaf;
	hData_ =hSamp;
	hRoot()->samps=points;		
	if( hForest->skdu.step==53 ){
		//name+="0";
		//hRoot()->ToBmp( (BMPP *)(user_data),0x0 );
	}
	GST_VERIFY( nSamp>0,"DecisionTree::Clasify nSampe is X" );
	hRoot()->Split( );	


}		

void DecisionTree::Regress( arrPFNO &points,int flag ){
	int nSamp=points.size( ),sum=0,nz,no;
	WeakLearners vLeaf;
	hRoot()->samps=points;			
	GST_VERIFY( nSamp>0,"DecisionTree::Regress nSampe is X" );
	hRoot()->Split( );	
}

void WeakLearner::AtLeaf( int flag ){
	FeatsOnFold *hData=hTree->GetDat( );
	int i,j,tag,no,nCls=hData->nCls,nSamp=samps.size( ),*tags= hData->Tag();
	GST_VERIFY( nSamp!=0,"WeakLearner::AtLeaf nSamp is 0" );
	if( hTree->hForest->isStage(RF_TRAIN) ){
		assert( distri==nullptr );
		distri=new float[nCls];
		for( i=0; i<nCls; i++ )		distri[i]=0;
		double s=1.0/nSamp;//samps.size( );
		for( arrPFNO::iterator it=samps.begin(); it!=samps.end();it++ ){
			F4NO *hfn=*it;				no=hfn->pos;
			tag= tags[no];			assert(tag>=0 && tag<nCls);
			distri[tag]+=1.0;
		}
		for( i=0; i<nCls; i++ )	{
			distri[i]*=s;
		}
	}else{
		assert( distri!=nullptr );
	}
	if( 1 ){		//累积到hData->distri
		float *dtr;
		for (auto hfn : samps) {
		//for each( F4NO *hfn in samps ){
			no=hfn->pos;		dtr=hData->distri+no*nCls;
			for( j=0;j<nCls;j++){
				dtr[j]+=distri[j];
			}
		}			
	}
}

bool WeakLearner::Model2Distri( int nCls,double *dist_,int flag ){
	assert( samps.size()==0 );
	int i,cls;
	for( i=0; i<nCls; i++ )		dist_[i]=distri[i];
	return true;
}


//bootstrap samples from the original data
void DecisionTree::BootSample( arrPFNO &boot,arrPFNO &oob,FeatsOnFold *hDat,int flag ){
	int total=hDat->nSample( ),i,j,cls,nCls=hDat->nCls,nSample=total,no,nz;
	assert(total>0);
	double sBalance=hForest->sBalance;
	int *info=hDat->info,*tag=hDat->Tag();
//	assert( toObj.size( )==total );
	if( sBalance>0.5 && sBalance<1.0 ){
		for( i=0;i<nCls;i++ ){
			if( hDat->nEach[i]==0 )
				continue;
			nSample=MIN(nSample,hDat->nEach[i]);
		}
		nSample*=sBalance;
	}
	hDat->Shuffle( );
	oob.clear( );			boot.clear();		
	//vector<int>nos;
	for( cls=0;cls<nCls;cls++ ){
		if( hDat->nEach[cls]==0 )
			continue;
		if( sBalance<0 ){
			nSample=hDat->nEach[cls]*0.66;
		}
		for( nz=0,j=0;j<total;j++  ) {
			no=hDat->permut[j];
			if( tag[no]!=cls )	continue;
			if( nz++<nSample )
				boot.push_back( new F4NO(-1,no,cls,info[no]) );
			else
				oob.push_back( new F4NO(-1,no,cls,info[no]) );
		}
		nz = boot.size( );		
		//assert( nz==nSample*(cls+1) );
	}
}

WeakLearner::~WeakLearner( )		{
	ClearSamps( );
	if( distri!=nullptr )
		delete[] distri;
}		


void WeakLearner::ToCPP( int flag )	{	hTree->hForest->ToCPP( this,flag );		}


void MT_BiSplit::Dump(const char* info_0, int type, int flag) {
	assert(fruit!=nullptr);
	double f= fruit->Thrshold(false);	//fruit.Get_<float>( );
	if (left == nullptr) {

	}	else {
		double l_down=left->GetDownStep(),r_down = right->GetDownStep();
		string ft= flag ==0 ? "BLIT":"FOLD";
		printf( "\t%s\tF=%d samp=%lld(%lld,%lld),fruit=%g impuri={%g,%g} down={%g,%g}\n", info_0,feat_id,nSample(),
			fruit->nLeft, fruit->nRight,f, left->impuri,right->impuri,l_down, r_down);

	}
}

