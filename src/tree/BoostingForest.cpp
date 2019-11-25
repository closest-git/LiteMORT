#include "./BoostingForest.hpp"
#include <random>
#include <thread>
#include <time.h>
#include <thread>
#include <stack>

//double WeakLearner::minSet=0.01;
//double WeakLearner::minGain=0.01;
static	char ssPath[256];		//only for debug

#if defined (_MSC_VER)  // Visual studio
    #define thread_local __declspec( thread )
#elif defined (__GCC__) // GCC
    #define thread_local __thread
#endif

void BoostingForest::VerifyTree( DecisionTree hTree,FeatsOnFold *hDat,int flag ){
/*	int nSamp=hDat->nSample( ),i,tag;
	double *distri=new double[nClass*2];
	for( i=0; i<nSamp; i++ ){
		float *vec=TO<float>(hDat,i);
		tag=BoostingForest::Clasify( vec,distri );
		assert( tag==hDat->tag[i]);
	}
	delete[] distri;*/
}

void BoostingForest::ErrEstimate( FeatsOnFold *hData,DForest& trees,int dataset,int flag ){	

}

void BoostingForest::TestOOB( FeatsOnFold *hData,int flag ){
	int nzLeaf=0,total=hData->nSample( );
	double a,sOOb=0;
	stage = RF_TEST;
	if( model==REGRESSION ){
		eOOB=0;		eInB=0;
		for (auto hTree : forest) {
		//for each( DecisionTree *hTree in forest )	{
			nzLeaf+=hTree->nLeaf;
			if( hTree->nLeaf==1 )
			{	printf( "\n>>>>>>RF_%s only 1 Leaf,Please check reason!!!",hTree->name.c_str() );	}
			assert( hTree->hData_==hData );
			if( hTree->oob.size( )==0 )	continue;
			sOOb+=hTree->oob.size( );
			hTree->Regress( hTree->oob );
			eOOB += ErrorAt( hTree->oob );
		}
		eOOB = sOOb==0.0 ? DBL_MAX : eOOB;		//Î´Ð£ÑéOOB
	}else{
		float *distri=hData->distri;
		hData->Distri2Tag( nullptr,nClass,0x0);
		eInB = 1.0-hData->rOK;	
		memset( distri,0x0,sizeof(float)*total*nClass );
		for (auto hTree : forest) {
		//for each( DecisionTree *hTree in forest )	{
			nzLeaf+=hTree->nLeaf;
			if( hTree->nLeaf==1 )
			{	printf( "\n>>>>>>RF_%s only 1 Leaf,Please check reason!!!",hTree->name.c_str() );	}
			assert( hTree->hData_ ==hData );
			sOOb+=hTree->oob.size( );
			hTree->Clasify( hData,hTree->oob,distri );
		}
		sOOb/=forest.size( );
		hData->Distri2Tag(nullptr, nClass, 0x0);
		eOOB = 1.0-hData->rOK;			hData->err=eOOB;
	}
	a = (total-sOOb)/(nzLeaf*1.0/forest.size());
//	printf( "\r%d(%d,%d)\toob=%g,pt@leaf=%.4g,err=(%3.2g,%3.2g)\n",skdu.step,total,curF.size(),sOOb,a,eInB*100,eOOB );
//	printf( "\r\noob=%g,pt@leaf=%.4g,err=(%4.3g%%,%4.3g%%)\n",sOOb,a,eInB*100,eOOB*100 );
}
void BoostingForest::Train( FeatsOnFold *hData,int flag ){
	if( hData==nullptr )
		hData=hTrainData;
	stage = RF_TRAIN;
	int nIns=0,no=0,total,i,j,nTree=forest.size();
	//FeatsOnFold *hData=curF[0]->hData;
	total=hData->nSample( );			assert(nClass>0);
	float *distri=hData->distri,*dtr=nullptr,tag,d1,rOK=0;	
	memset( distri,0x0,sizeof(float)*total*nClass );
	WeakLearner* hWL=nullptr;
	impurity=0;
	double a,sOOb=0;
//	DForest curF;

#pragma omp parallel for num_threads(nThread) private( i ) 
	for ( i=0;i<nTree;i++ )	{
		DecisionTree* hTree = forest[i];
		skdu.noT=no++;			//printf( "\nTree_%d...",no );
		if( hTree->isTrained(flag) ||hTree->hData_ !=hData )
			continue;
		hTree->Train( flag );
//		curF.push_back( hTree );
	}
	AfterTrain( hData,flag );
	
	//std::sort( ginii.begin(),ginii.end(),F4NO::isSmall );
//	curF.clear( );
}

void BoostingForest::Clasify( int nSamp,FeatsOnFold *hSamp,int *cls,int flag ){
	stage = RF_TEST;

	int i,j,nz,sum=0,tag,no;
	WeakLearners vLeaf;
	float *distri=new float[nSamp*nClass](),d1,*dtr=nullptr,*model;
	arrPFNO points;
	for( i=0;i<nSamp;i++ )	{
		points.push_back( new F4NO(-1.0,i) );
	}
	for (auto hTree : forest) {
	//for each( DecisionTree *hTree in forest ){
		hTree->Clasify( hSamp,points,distri );	
	}
	for (auto hfn : points) { delete hfn; }
	//for each( F4NO *hfn in points ){	delete hfn;	}
	points.clear( );
	dtr=distri;
	for(i=0;i<nSamp;i++,dtr+=nClass){
		for( tag=-1,d1=0,j=0;j<nClass;j++ ){
			if(dtr[j]>d1)
			{	d1=dtr[j];	tag=j;	}
		}
		assert( tag>=0 && tag<nClass );
		cls[i]=tag;
	}
	delete[] distri;
}
void BoostingForest::Clasify( int nSamp,FeatsOnFold *hSamp,int flag ){
	stage = RF_TEST;
	GST_VERIFY( nSamp<=hSamp->nSample(),"" );
	int i,j,nz,sum=0,tag,no;
	float *distri=hSamp->distri,d1,*dtr=nullptr,*model;
	memset( distri,0x0,sizeof(float)*nSamp*nClass );
	arrPFNO points;
	for( i=0;i<nSamp;i++ )	{
		points.push_back( new F4NO(-1.0,i) );
	}
	for (auto hTree : forest) {
	//for each( DecisionTree *hTree in forest ){
		hTree->Clasify( hSamp,points,distri );	
	}
	for (auto hfn : points) { delete hfn; }
	//for each( F4NO *hfn in points ){	delete hfn;	}
	points.clear( );
	hSamp->Distri2Tag(nullptr, nClass, 0x0);
	printf( "\n\r%s rOK=%d:%3.2g\n",name.c_str(),nSamp,hSamp->rOK*100 );
}

void BoostingForest::InitFeat( int type,int nFlag ){
	for (auto feat : hTrainData->feats) {
		if (BIT_TEST(feat->type, FeatVector::REPRESENT_)) {
			
		}
	}
}


void BoostingForest::SetUserData( void*ud_,isTrueObj hf,int flag ){
	assert( ud_!=nullptr && hf!=nullptr );
	hfIsObj=hf;
	user_data=ud_;
}

void BoostingForest::ClearData( )	{
	//printf("********* ClearData hTrainData=%p,hTestData=%p,hEvalData=%p", hTrainData, hTestData,hEvalData);
	if (hTrainData != nullptr) {
		delete hTrainData;		hTrainData = nullptr;
	}
	if (hTestData != nullptr) {
		delete hTestData;		hTestData = nullptr;
	}
	if (hEvalData != nullptr) {
		delete hEvalData;		hEvalData = nullptr;
	}
	//for each( CASE* hCase in SamplSet )		delete hCase;
	//SamplSet.clear( );
}

void BoostingForest::Clear( ){
	if (prune != nullptr) {
		delete prune;		prune = nullptr;
	}
	ClearHisto();
	/*if (histo_buffer != nullptr)	{
		delete histo_buffer;		histo_buffer = nullptr;
	}*/
	if (isRefData) {

	}	else {
		ClearData( );
	}

	WeakLearners vNodes;
	for (auto hTree : forest) {
	//for each( DecisionTree *hTree in forest ){
		delete hTree;	
	}
	forest.clear( );
}





