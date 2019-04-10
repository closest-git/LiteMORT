#include "stdafx.h"
#include "RF_ConfiRegress.h"
#include "GruST\util\BLAS_t.hpp"

#define _BIT_FEAT_MAX_ 1000
#define _BIT_FEAT_a0_ -0.1
#define _BIT_FEAT_a1_ 0.1
#define _BIT_FEAT_TO_SAVE_SPACE_

void RF_ConfiRegress::AfterTrain( int cas,int nMulti,int flag ){
	return;
}

void RF_ConfiRegress::DumpTree( int xno,DecisionTree *hTree,int flag ){
	WeakLearners vLeaf;
	hTree->GetLeaf( vLeaf );
	std::sort( vLeaf.begin(),vLeaf.end(),WeakLearner::isBig );
	printf( "%d leafs\n",vLeaf.size() );
	double e0=DBL_MAX,e1=0;
	int no=0,nz=0,i;
	for each( WeakLearner* hWeak in vLeaf ){
		for( i=0;i<hWeak->nLastSamp;i++ ){
		}
		nz+=hWeak->nLastSamp;		
		e1=MAX(e1,hWeak->err);			e0=MIN(e0,hWeak->err);
		printf( "%4.2g(%3.2g)@%d, ",hWeak->err,hWeak->impuri,hWeak->nLastSamp );
	}
	printf( "\t	---------- [%g-%g] @%d samps\n",e0,e1,nz );
}

double RF_ConfiRegress::ErrorAt( arrPFNO& samps ){
	int no,nSamp=samps.size(),nMark=Trains[0]->vS.rows();	assert(nSamp>0);
	double err=0.0,a;
	for each ( F4NO *fn in samps  ){
		no=fn->pos;			assert(no>=0 && no<hTrainData->ldF );
		ShapeBMPfold* hFold=Trains[no];
		//err+=InterocErr(Trains[no])/nSamp;	//Trains[no]->vR.squaredNorm( );
		err += hFold->NormalErr( )/nSamp;
	}
	//err = sqrt( err/nSamp );
	return err;
}
void RF_ConfiRegress::Confi_Impuri(  WeakLearner *hWeak,int flag ){
	RandRegress *hRegress=dynamic_cast<RandRegress *>(hWeak);
	GST_VERIFY( hRegress!=nullptr,"RF_ConfiRegress::Impurity" );
	int no,nSamp=hWeak->samps.size(),i=0;
	//ShapeBMPfold::VECT errV,meanV;
	double impuri=0.0,a1=0,a2=0,*confi=new double[nSamp];
	for each ( F4NO *fn in hWeak->samps  ){
		no=fn->pos;			assert(no>=0 && no<hTrainData->ldF );
		a1=1.0-Trains[no]->NormalErr();	
		confi[i++]=a1;		
	}
	double mean,devia,sigma=0,sigmaL,sigmaR;
	STA_distribute( nSamp,confi,mean,devia );		
	hRegress->impuri=impuri=devia*devia*nSamp;
	assert( hRegress->sum.size()==0 );
	hWeak->err=hRegress->impuri;		
	hWeak->confi = mean;
	return ;
}

bool RF_ConfiRegress::LeafModel(  WeakLearner *hWeak,int flag ){
	int no,nSamp=hWeak->samps.size( );
	RandRegress *hRegress=dynamic_cast<RandRegress *>(hWeak);
	GST_VERIFY( hRegress!=nullptr,"" );
	ShapeBMPfold *hX=nullptr;
	double a,b,sum=0,a2=0.0;
	if( stage == RF_TRAIN ){		
		for each ( F4NO *fn in hWeak->samps   ){
			no=fn->pos;			assert(no>=0 && no<hTrainData->ldF );
			a = 1.0-Trains[no]->NormalErr( );		a2+=a*a;
			b = a-hWeak->confi;
			sum+=b*b;
		}
		sum=sqrt(sum/nSamp);		a2=sqrt(a2/nSamp);
		//if( skdu.noT%4==0 && skdu.noLeaf%4==0 )
		//	printf( "%d(%4.3g,%3.2g)\t",skdu.noLeaf,sum/a2,a2 );
	}else{
	}

	for each ( F4NO *fn in hWeak->samps   ){
		no=fn->pos;				hX=Trains[no];
		hX->confi += hRegress->confi;
		hX->nBag++;
	}
	return true;
}

using namespace Eigen;
bool RF_ConfiRegress::L_Regress( WeakLearner *hWeak,int flag ){
//	if(skdu.noLeaf%4!=0)	return false;

	int no=hWeak->samps[0]->pos;
	ShapeBMPfold *hX=Trains[no];
	int nSamp=hWeak->samps.size( ),dim=0,i=0,j,nReg=hX->regressor.size();
	nReg=hX->vMomen.size( );
	assert( nReg>=1 );
	double a,b,a1=0,nrmB,err=0,err1,beta0,beta1;
	RandRegress *hRegress=dynamic_cast<RandRegress *>(hWeak);
	GST_VERIFY( hRegress!=nullptr,"" );
	
	Eigen::MatrixXd mA(nSamp,nReg+1);
	Eigen::VectorXd rhs(nSamp),beta;
	for each ( F4NO *fn in hWeak->samps   ){
		no=fn->pos;			hX=Trains[no];
		mA(i,0)=1;
//		GST_VERIFY( hX->regressor.size()==nReg );
		float *move=hX->vMomen.data();
		for(j=0;j<nReg;j++ ){
			//mA(i,j+1)=a=hX->regressor[j];	//fn->val;		
			mA(i,j+1)=a=move[j];
		}
		//;mA(i,1)=fn->val;	
		rhs(i)=b=1.0-hX->NormalErr( );
		i++;
	}
	double mean,devia,x_0,x_1,nrm2,*rh=rhs.data();
	STA_distribute( nSamp,rhs.data(),mean,devia,x_0,x_1,nrm2 );
	printf( "%d:%d\t<%3.2g,%3.2g,%3.2g,%3.2g>=>(",skdu.noT,skdu.noLeaf,mean,devia,x_0,x_1 );

	beta = mA.jacobiSvd(ComputeThinU | ComputeThinV).solve(rhs);	
	for(j=0;j<=nReg;j++ ){		printf( "%3.2g,",beta[j] );	}
	beta0 = beta(0);		beta1 = beta(1);		
	/*for( i=0;i<nSamp;i++ ){
		a = rh[i]-(beta0+beta1*mA(i,1));		b=rhs(i);
		a = abs(a/b);
		err+=a;		a1 = MAX(a,a1);
	}
	err/=nSamp;*/
	rhs-=mA*beta;
	STA_distribute( nSamp,rhs.data(),mean,devia,x_0,x_1,nrm2 );
	for( mean=0,i=0;i<nSamp;i++ ){
		a=abs(rh[i]);		mean+=a;
	}
	mean/=nSamp;
	printf( ")\t<%3.2g,%3.2g,%3.2g,%3.2g>\n",mean,devia,x_0,x_1 );
//	b=rhs.squaredNorm()/nSamp;
//	printf( "R(%g-%g),nrmB=%g\t",err,b,nrmB );
	 
	return true;
}

double RF_ConfiRegress::LR( int nPt, Eigen::MatrixXd &mA0,Eigen::VectorXd &rhs0,Eigen::VectorXd &beta,int flag ){
	double a,b,a1=0,nrmB,err=0,err1,beta0,beta1;
	int nReg=mA0.cols(),i;
	assert( nPt<=rhs0.rows());
	Eigen::MatrixXd mA=mA0.block(0,0,nPt,nReg).eval();			
	Eigen::VectorXd rhs=rhs0.head(nPt).eval();
	double mean,devia,dev0,x_0,x_1,nrm2,*rh=rhs.data();
	STA_distribute( nPt,rh,mean,dev0,x_0,x_1,nrm2 );
	//printf( "%d:%d\t<%3.2g,%3.2g,%3.2g,%3.2g>=>(",skdu.noT,skdu.noLeaf,mean,devia,x_0,x_1 );

	beta = mA.jacobiSvd(ComputeThinU | ComputeThinV).solve(rhs);	
	//for(i=0;i<nReg;i++ ){		printf( "%3.2g,",beta[i] );	}
	beta0 = beta(0);		beta1 = beta(1);		
	/*for( i=0;i<nSamp;i++ ){
		a = rh[i]-(beta0+beta1*mA(i,1));		b=rhs(i);
		a = abs(a/b);
		err+=a;		a1 = MAX(a,a1);
	}
	err/=nSamp;*/
	rhs-=mA*beta;
	STA_distribute( nPt,rhs.data(),mean,devia,x_0,x_1,nrm2 );
	for( mean=0,i=0;i<nPt;i++ ){
		a=abs(rh[i]);		mean+=a;
	}
	mean/=nPt;
	//printf( ")\t<%3.2g,%3.2g,%3.2g,%3.2g>\n",mean,devia,x_0,x_1 );
	return devia*devia*nPt;
}
hBLIT RF_ConfiRegress::GetBlit( WeakLearner *hWeak,int flag ){
	if( hWeak->hBlit==nullptr ){
		hWeak->hBlit=new BLIT_Diff( );		
	}
	RandRegress *hRegress=dynamic_cast<RandRegress *>(hWeak);
	GST_VERIFY( hRegress!=nullptr,"RF_ConfiRegress::GetBlit" );
	BLIT_Diff *hBlit=dynamic_cast<BLIT_Diff*>(hWeak->hBlit);
	int i,j,nPt=hTrainData->nFeat,nSamp=hWeak->samps.size(),nRand=this->nPickAtSplit( nullptr ),nCand=cands.size();
	int *iUs=new int[nRand](),*iVs=new int[nRand]( ),iU,iV;
	std::uniform_real_distribution<float> uniFloat(0,1);
	float *thrs=new float[nRand](),*confi=new float[nSamp];
	//double *confi=new double[nSamp*3],*confL=confi+nSamp,*confR=confL+nSamp;
	Eigen::MatrixXd aL(nSamp,2),aR(nSamp,2);
	Eigen::VectorXd rhsL(nSamp),rhsR(nSamp),betL(2),betR(2);
	i=0;
	for each ( F4NO *fn in hWeak->samps  ){
		no=fn->pos;			assert(no>=0 && no<hTrainData->ldF );
		confi[i++] = 1.0-Trains[no]->NormalErr( );
	}
	double mean,devia,sigma=0,sigmaL,sigmaR;
	STA_distribute( nSamp,confi,mean,devia );		sigma=devia*devia*nSamp;
	for( i=0; i<nRand; i++ ){		//预先生成，避免多线程的不确定性
		do{
			iU=UniformInt(0,nPt-1);		iV=UniformInt(0,nPt-1);
			if( cand_dis[iU*nCand+iV]>uniFloat(*hRander) )
			{	iUs[i]=iU,	iVs[i]=iV;		break;		}
		}while( 1 );
		//		{	featLines.push_back( new F4NO(-1,-1,iU,iV) );			}

		float *fu=hTrainData->Feat(iU),*fv=hTrainData->Feat(iV);
		no = UniformInt(0,nSamp-1);				no=hWeak->samps[no]->pos;		
		float th1=fu[no]-fv[no],th2;	
		no = UniformInt(0,nSamp-1);				no=hWeak->samps[no]->pos;		
		th2=fu[no]-fv[no];
		thrs[i]=(int)((th1+th2)/2.0+0.5);
	}
	ShapeBMPfold *hX=nullptr;
	assert( Trains.size()==hTrainData->ldF );
	nzWeak+=nRand;
//#pragma omp parallel for num_threads(nBlitThread) private( i ) 
	for( i=0; i<nRand; i++ ){
		float a;
		int iU=-1,iV=iU,nLeft=0,nRight=0,no;	
		iU=iUs[i];			iV=iVs[i];	
		float eL,eR,one=1.0,thrsh=thrs[i];
		float g,*fu=hTrainData->Feat(iU),*fv=hTrainData->Feat(iV);	
		for each ( F4NO *fn in hWeak->samps  ){
			no=fn->pos;			assert(no>=0 && no<hTrainData->ldF );
			a = 1.0-Trains[no]->NormalErr( );
			if( fu[no]-fv[no]<=thrsh ){
		//	if( fu[no]<=thrsh ){
				aL(nLeft,0)=1.0;				aL(nLeft,1)=fu[no]-fv[no];	
				rhsL(nLeft)=a;					nLeft++;
			}else{
				aR(nRight,0)=1.0;				aR(nRight,1)=fu[no]-fv[no];	
				rhsR(nRight)=a;					nRight++;
			}
		}	
		assert( nRight+nLeft==nSamp );
		if( nLeft<WeakLearner::minSet || nRight<WeakLearner::minSet )	
		{	continue;	}			
		//STA_distribute( nLeft,confL,mean,devia );		sigmaL=devia*devia*nLeft;
		//STA_distribute( nRight,confR,mean,devia );		sigmaR=devia*devia*nRight;


		sigmaL=LR(nLeft,aL,rhsL,betL);						
		sigmaR=LR(nRight,aR,rhsR,betR);
		g = sigma-sigmaL-sigmaR;		assert(g>-FLT_EPSILON);
		if( g>hBlit->gain ){
#pragma omp critical
			if( g>hBlit->gain ) {	
				hBlit->id = iU;		hBlit->other = iV;	hBlit->thrsh=thrsh;		hBlit->gain=g;	
				printf( "gain(%g)-<%d,%d,%g> left=(%g,%d) rigt=(%g,%d)\n",g,iU,iV,thrsh,sigmaL,nLeft,sigmaR,nRight );
			}		
		}
	}	
	delete[] iUs;		delete[] iVs;			delete[] thrs;
	delete[] confi;
	if(0){	//仅用于测试
		int nLeft=0,nRight=0;
		GetFeatDistri( hWeak );
		for each ( F4NO *fn in hWeak->samps  )	{
			if( fn->val<=hBlit->thrsh ){	
				nLeft++;
			}else
				nRight++;
		}	
		assert( nLeft>=WeakLearner::minSet && nRight>=WeakLearner::minSet );	
	}
		
	return hBlit;
}


void RF_ConfiRegress::AfterTrain( FeatData *hData,int flag ){
	if( isDumpLeaf )	{
		SHAPE_IMAGE si(96,96,3);		si.bpp=24;
		for each( DecisionTree *hTree in forest )	{	
			PICS_N pics;
			DumpTree( skdu.step,hTree );
			WeakLearners vLeaf;
			hTree->GetLeaf( vLeaf );
			for each( WeakLearner* hWL in vLeaf ){
				for each( F4NO *fn in hWL->samps ){
					ShapeBMPfold *hSamp=Trains[fn->pos];
				//	pics.nails.push_back( hSamp->TraceBmp("",si,0x0) );
				}
				hWL->nLastSamp=hWL->samps.size();		hWL->ClearSamps( );
			}
		//	pics.PlotNails( sPath.c_str(),si,ldPic,true );
		}
	}
}

void RF_ConfiRegress::OnMultiTree( int cas,int nMulti,int flag ){
	double oob0=0,err=0,im1=0,a,b,a0=DBL_MAX,a1=-DBL_MAX,b0=DBL_MAX,b1=-DBL_MAX;
	int total=Trains.size(),nBoot=total-nOOB,no=0,nzBag=0,bag0=0;
	eOOB=0;
	for each( ShapeBMPfold *hFold in Trains ){
		b = 1.0-hFold->NormalErr( );	
		if( hFold->nBag>0 ){
			hFold->confi/=hFold->nBag;
			G_MINMAX(b,b0,b1);
		}	else		{	
			bag0++;			continue;
		}
		a = hFold->confi-b;		G_MINMAX(a,a0,a1);
		if( no<nBoot ){		//
			err+=a*a;		nzBag+=hFold->nBag;
			assert( hFold->nBag<=nMulti );
		}else{				//oob set
			eOOB += a*a;
			assert( hFold->nBag==nMulti );
		}
		hFold->nBag=0;
		no++;
		hFold->confi=0.0;
	}
	err=sqrt(err/nBoot);
	eOOB = nOOB>0 ? sqrt(eOOB/nOOB) : DBL_MAX;
//	printf( "%d: err=(%5.3g,%5.3g) eOOB=%g nTree=%d nWeak=%g,eta=%g\n",step,im1,hRoot->err,eOOB,forest.size(),nzWeak/(step+1),eta );
	printf( "\n********* ConfiRegress:: err=(%4.3g,%4.3g,%4.3g) b=(%4.3g,%4.3g),bag=%g(%d) eOOB=%g->%g @%d\n",err,a0,a1,b0,b1,
		nzBag*1.0/nBoot,bag0,oob0,eOOB,nOOB );	
}

RF_ConfiRegress::RF_ConfiRegress( RF_ShapeRegress *hReg_ ) : hRegress(hReg_){
	//boot=BT_ALL;
	boot=hReg_->boot;
	model=REGRESSION;			regular = MULTI_TREE ;
	GST_VERIFY( hRegress!=nullptr,"RF_ConfiRegress::RF_ConfiRegress" );

	isToCPP=false;
	index=hReg_->index;			GST_VERIFY(index==ShapeBMPfold::FOLD_MAP,"RF_ConfiRegress" );
	std::copy( hRegress->Trains.begin(),hRegress->Trains.end(),std::back_inserter(Trains) );
	nPickWeak=hReg_->nPickWeak;
	skdu = hReg_->skdu;			skdu.nTree=10;
	cand_dis = hReg_->cand_dis;
	maxDepth = hReg_->maxDepth;
	maxDepth = 8;
	InitRander( skdu.cascad+314159 );
	nTree=skdu.nTree;
	int nTrain=Trains.size( ),nTest=Tests.size(),nz=0,nCand=hReg_->cands.size();		
	hTrainData=new FeatData( nullptr,nTrain,nCand,1,0 );	
	eta=0.0;
	lenda=0.0;

	nBlitThread=hReg_->nBlitThread;
	if( nBlitThread>1 ){
		arrPTF=new ShapeBMPfold::PTFs[nBlitThread];
		for( int i=0; i<nBlitThread;i++ )	{
			arrPTF[i].resize( nCand );
			std::copy( hReg_->cands.begin(),hReg_->cands.end(),arrPTF[i].begin() );
		}
	}
}

int RF_ConfiRegress::Train( string sTitle,int cas,int flag )	{
	GST_TIC(tick);	
	
	eOOB = 1.0;
	//skdu.step++;		//rand_seed_0=RAND_REINIT;
	int nSample=Trains.size( ),k;
	UpdateFeat( );
	//hTrainData->Reshape( );
	assert( forest.size()==0 );	

	double im0=0,im1=0,a;
	for each( ShapeBMPfold *hFold in Trains ){
		hFold->regressor.clear( );
		hFold->UpdateErr( );	
		im0+=hFold->vR.squaredNorm( );
	}
	if( im0/nSample<FLT_EPSILON )
	{	eOOB=0;	printf( "\n********* You are LUCKY *********\n");	return 0x0;	}
	
	printf( "\n********* ConfiRegress::%s_%d(%d-%d,im=%g),nOOB=%d,nTree=%d,boot=%d,dep1=%d,eta=%g,lenda=%g,weak=%d minSet=%g...\r\n",
		sTitle.c_str(),cas,Trains.size()-nOOB,nOOB,sqrt(im0/nSample),nOOB,nTree,boot,maxDepth+1,
		eta,lenda,nPickAtSplit(nullptr),WeakLearner::minSet );

	nzWeak=0.0;
	DForest curF;
	if( isToCPP	)		ToCPP( cas,0x0 );
	double eta0=eta/2,eta1=eta*2,et=(eta1-eta0)*2/nTree;
	int t,noT=0;
	for( skdu.step=0; skdu.step<skdu.nTree; skdu.step++,noT++){
		skdu.noT=noT;
		DecisionTree *hTree=new DecisionTree( this,hTrainData );			hTree->name=to_string(666)+"_"+to_string(noT);	
		WeakLearner* hRoot=hTree->hRoot( );
		im1 = hRoot->impuri;
		forest.push_back( hTree );
		//isDumpLeaf = noT==nTree-1;
		RandomForest::Train( hTrainData );
		TestOOB( hTrainData );
		
		k=0;
		for each( DecisionTree *hTree in forest )	{		
			delete hTree;	
		}
		forest.clear( );		
	}
	OnMultiTree( cas,skdu.nTree );
	AfterTrain( cas,skdu.nTree );
	if( isToCPP	)	{	
		fprintf( fpC,"\n}\n"		);
	}
//	printf( "********* %s time=%g\r\n\n",sTitle.c_str(),GST_TOC(tick) );
	
	//printf( "\"RF\":%s(%d) nPt=%d nFeat=%d minGain=%g minSet=%g\n",name.c_str(),nTree,nPt,nFeat,WeakLearner::minGain,WeakLearner::minSet );
	return 0x0;
}
