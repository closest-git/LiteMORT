#include "stdafx.h"
#include "RF_ShapeRegress.h"
#include "RF_ConfiRegress.h"
#include "GruST\util\BLAS_t.hpp"

#define _BIT_FEAT_MAX_ 1000
#define _BIT_FEAT_a0_ -0.1
#define _BIT_FEAT_a1_ 0.1
#define _BIT_FEAT_TO_SAVE_SPACE_
//参加randomly_generate_split_feature的精巧实现
void RF_ShapeRegress::RandCandidate( int nCand,ShapeBMPfold &mean,ShapeBMPfold::PTFs &cands,int flag ){
	if( index==ShapeBMPfold::INTER_2 )
	{	RandCandidate_2( nCand,mean,cands,flag );	return ;}
	cands.clear( );		cands.resize( nCand );
	//Eigen::VectorXd colX=mean.vY.col(0),colY=mean.vY.col(1);
	ShapeBMPfold::VECT colX=mean.vY.col(0),colY=mean.vY.col(1);
	double x0=colX.minCoeff(),x1=colX.maxCoeff(),y0=colY.minCoeff(),y1=colY.maxCoeff();
//	std::uniform_real_distribution<double> distX(x0,x1),distY(y0,y1);
	std::uniform_real_distribution<double> distX(-1,1),distY(-1,1);
	int i,j,nPt=cands.size( ),no;
	float *vX=new float[nPt],*vY=new float[nPt],dis=0,*inteval=new float[nPt+1],*weight,sum;
	for( i = 0; i < nPt;i ++ ){
		ShapeBMPfold::PTonF &pt=cands[i];		pt.indx=index;
		vX[i]=distX(*hRander);			vY[i]=distY(*hRander);
		//assert( vY[i]>=y0 && vY[i]<=y1 );
		dis+=mean.IndexPt( vX[i],vY[i],pt );
	}
	dis /= nPt;
	if( 0 ){
		SHAPE_IMAGE si(128,128,3);		si.bpp=24;
		GST_BMP bmp( &si );
		mean.SetROI( RoCLS( 16,16,96,96 ) );
		mean.PtsOnFold( cands,mean.vY );
		for each ( ShapeBMPfold::PTonF pt in  cands ){
			bmp.SetPixelColor( pt.gc,pt.gr,255,0,0 );
		};
		bmp.Save( "F:\\GiFace\\trace\\cand_"+to_string(flag)+".bmp" );
	}
	//arrPDis.clear( );
	if( lenda>0 ){
		for( i=0;i<nPt;i++ ){
			x0 = vX[i];			y0 = vY[i];
			weight = cand_dis+i*nCand;
			for( sum=0,j=0;j<nPt;j++ ){
				x1 = vX[j];		y1 = vY[j];
				dis = sqrt((x0-x1)*(x0-x1)+(y0-y1)*(y0-y1));		sum+=dis;
				weight[j] = (j==i) ? 0 : exp(-dis*lenda);
				assert( weight[j] >=0 && weight[j] <=1 );
				inteval[j]=j;
			}
			sum/=nPt;
		/*	inteval[nPt]=nPt;
			std::piecewise_constant_distribution<float> dist(inteval,inteval+nPt,weight );
			arrPDis.push_back( dist );
			if( 0 ){
				for( sum=0,j=0;j<1000;j++ ){
					no=(int)(dist(*hRander));		
					x1 = vX[no];		y1 = vY[no];
					dis = sqrt((x0-x1)*(x0-x1)+(y0-y1)*(y0-y1));			sum+=dis;
				}
				sum/=1000;
			}*/
		}	
	}
	
	delete[] inteval;		//delete[] weight;		
	delete[] vX;			delete[] vY;
}

void RF_ShapeRegress::RandCandidate_2( int nCand,ShapeBMPfold &mean,ShapeBMPfold::PTFs &cands,int flag ){
	cands.clear( );		cands.resize( nCand );
	//Eigen::VectorXd colX=mean.vY.col(0),colY=mean.vY.col(1);
	ShapeBMPfold::VECT colX=mean.vY.col(0),colY=mean.vY.col(1);
	double x0=colX.minCoeff(),x1=colX.maxCoeff(),y0=colY.minCoeff(),y1=colY.maxCoeff();
	int i,j,no,iU,iV,nMark=mean.vY.rows( );
	std::uniform_int_distribution<int> uniti(0,nMark-1);
	float *vX=new float[nCand],*vY=new float[nCand],dis=0,*weight,sum;
	ShapeBMPfold::VECT &fold=mean.vY;
	for( i = 0; i < nCand;i ++ ){
		ShapeBMPfold::PTonF &pt=cands[i];		pt.indx=ShapeBMPfold::INTER_2;
		pt.iU=iU=uniti(*hRander);		assert( pt.iU>=0 && pt.iU<nMark );
		do{		pt.iV=uniti(*hRander);		}		while( pt.iU==pt.iV );	
		vX[i]=(fold(pt.iU,0)+fold(pt.iV,0))/2;
		vY[i]=(fold(pt.iU,1)+fold(pt.iV,1))/2;
	}
	if( 0 ){
		SHAPE_IMAGE si(128,128,3);		si.bpp=24;
		GST_BMP bmp( &si );
		mean.SetROI(RoCLS( 16,16,96,96 ));
		mean.PtsOnFold( cands,mean.vY );
		for each ( ShapeBMPfold::PTonF pt in  cands ){
			bmp.SetPixelColor( pt.gc,pt.gr,255,0,0 );
		};
		bmp.Save( "F:\\GiFace\\trace\\cand_"+to_string(flag)+".bmp" );
	}
	//arrPDis.clear( );
	if( lenda>0 ){
		for( i=0;i<nCand;i++ ){
			x0 = vX[i];			y0 = vY[i];
			weight = cand_dis+i*nCand;
			for( sum=0,j=0;j<nCand;j++ ){
				x1 = vX[j];		y1 = vY[j];
				dis = sqrt((x0-x1)*(x0-x1)+(y0-y1)*(y0-y1));		sum+=dis;
				weight[j] = (j==i) ? 0 : exp(-dis*lenda);
				assert( weight[j] >=0 && weight[j] <=1 );
			}
			sum/=nCand;
		/*	inteval[nPt]=nPt;
			std::piecewise_constant_distribution<float> dist(inteval,inteval+nPt,weight );
			arrPDis.push_back( dist );
			if( 0 ){
				for( sum=0,j=0;j<1000;j++ ){
					no=(int)(dist(*hRander));		
					x1 = vX[no];		y1 = vY[no];
					dis = sqrt((x0-x1)*(x0-x1)+(y0-y1)*(y0-y1));			sum+=dis;
				}
				sum/=1000;
			}*/
		}	
	}
	delete[] vX;			delete[] vY;
}

//为何BT_RANDOM_3，BT_MAX_ERR没效果		1/31/2016	cys
//BT_MIN_ERR 效果最好，有意思。似乎err较大的例子其实是错误的

void RF_ShapeRegress::BootSample( DecisionTree *hTree,arrPFNO &inb,arrPFNO &oob,FeatData *hDat,int flag ){
	string sBoot="ALL";
	int total=hDat->nSample( ),i,no,nz=0,nBoot=total-nOOB,isDump=Tests.size()==0;
	assert(total>0 );
	for each( F4NO *fn in Tests ) delete fn;
	Tests.clear( );		oob.clear( );
	for(i=0;i<total;i++){
		no=i;
		ShapeBMPfold *hX=Trains[no];
		double err=hX->NormalErr();
		if( nz++<nBoot )
			inb.push_back( new F4NO(err,no) );
		else{
			Tests.push_back( new F4NO(-1,no) );
			oob.push_back( new F4NO(-1,no) );
		}
	}
	if( boot==BT_RANDOM_3 ){
		sBoot="RANDOM_";
		std::random_shuffle( inb.begin(),inb.end() );
	}else if( boot==BT_MAX_ERR  ){
		sBoot="MAX_";
		std::sort( inb.begin(),inb.end(),F4NO::isPBig );
	}else if( boot==BT_MIN_ERR  ){
		sBoot="MIN_";
		std::sort( inb.begin(),inb.end(),F4NO::isPSmall );
	}
	if( boot!=BT_ALL ){
		nBoot = (int)(inb.size( )*2.0/3);
		inb.resize( nBoot );
	}
	//if( isDump )	
	//	printf( "********* BOOT-%s (%d-%d) \n",sBoot.c_str(),inb.size(),oob.size() );
}

void RF_ShapeRegress::AfterTrain( int cas,int nMulti,int flag ){
	for each( F4NO *fn in Tests ){
		no=fn->pos;
		ShapeBMPfold *hX=Trains[no];
		assert( fn->val==-1.0 );
		fn->val=hX->NormalErr( );
	}
	std::sort( Tests.begin(),Tests.end(),F4NO::isPSmall );
	int nTest=Tests.size( ),nSec=10,sec=(int)ceil(nTest*1.0/nSec),i,j,i_0=0,i_1,ldPic = 10;
	if( sec<0 )		return;

	PICS_N pics;
	SHAPE_IMAGE si(96,96,3);		si.bpp=24;
	string sNailPath="F:\\GiFace\\trace\\oob_"+to_string(cas)+".bmp";
	double err=0;
	printf( "********* OOB:[%g-%g]\t",Tests[0]->val,Tests[nTest-1]->val );
	for( i=0;i<nSec;i++,i_0+=sec ){
		i_1=MIN(i_0+sec,nTest );
		for( err=0,j=i_0;j<i_1;j++ )		{
			F4NO *fn=Tests[j];
			err+=fn->val;			
			if( i==nSec-1 )		{	//last bad sec
				ShapeBMPfold *hSamp=Trains[fn->pos];
				pics.nails.push_back( hSamp->TraceBmp("",si,0x0) );
			}
		}
		err /=(i_1-i_0);
		printf( "%g ",err );
	}	
	printf( "\n" );
	pics.PlotNails( sNailPath.c_str(),si,ldPic,true );
	for( err=0,j=0;j<10;j++ )		{		//best 10
		F4NO *fn=Tests[j];
		ShapeBMPfold *hSamp=Trains[fn->pos];
		printf( "%s-%g ",hSamp->nam.c_str(),fn->val );
	}
	printf( "\n" );	
}

void RF_ShapeRegress::UpdateFeat( int flag ){
	vector<int> vChn={
	//	GST_BMP::R,GST_BMP::G,GST_BMP::B,
		GST_BMP::L,GST_BMP::U,GST_BMP::V,
	};
	int i,nz=0,nTrain=Trains.size( ),nThread=nBlitThread,nCand=cands.size( );
	//for each ( ShapeBMPfold* hFold in Trains ){
//	ShapeBMPfold::PTFs *arrPTF=new ShapeBMPfold::PTFs[nThread];
	Eigen::RowVector2f off;
	Eigen::MatrixXf trans;	
#pragma omp parallel for num_threads(nBlitThread) private( nz ) 
	for( nz=0;nz<nTrain;nz++ ){
		int thred=omp_get_thread_num( );
		ShapeBMPfold* hFold=Trains[nz];
		if( index==ShapeBMPfold::SIMILAR_TRANS )
			hFold->SimilarRS( spMean.vY,trans,off,0x0 );		//真奇怪
	//	hFold->UpdateErr( );
		float *feat=TO<float>( hTrainData,nz );	
		hFold->GetFeat( vChn,arrPTF[thred],feat );	
		//hFold->TraceBmp("F:\\GiFace\\trace\\fold.bmp",SHAPE_IMAGE(),0x0);
		hTrainData->tag[nz]=0;
	}
//	delete[] arrPTF;
	hTrainData->Reshape( );
}

RF_ShapeRegress::RF_ShapeRegress(vector<ShapeBMPfold*>&trs,ShapeBMPfold &spM_,int nCand,int nStep_,int nEach,int nOB,int cas,int flag) : 
	RandomForest( ),Trains(trs),mOff(spM_.vY),mSum(spM_.vY),nOOB(nOB),spMean(spM_),regular(SINGLE_TREE),boot(BT_MIN_ERR),
	isToCPP(false),fpC(NULL),fpD(NULL),fpT(NULL),arrPTF(nullptr){
#ifdef _DEBUG
#endif
	model=RandomForest::REGRESSION;		
	skdu.cascad = cas;			skdu.nStep=nStep_;		
	int seed=cas+31415927;
	nTree=skdu.nStep*nEach;		skdu.nTree=nTree;
	regular = nEach>1 ? MULTI_TREE : SINGLE_TREE;
	isToCPP=true;
	InitRander( seed );
	//lenda=3.0;		//强烈overfit		
	lenda=10;	//MAX(1.0,10-cas*0.5);			
	nCand = 400;	//有意思
	eta = 0.1;
	nBlitThread=8;	//并行导致结果不再可重复
	//nBlitThread=1;	
	maxDepth=4;		//有意思,为啥取5无效果?
	sBalance=1.0;		
	model=REGRESSION;		
	//index=ShapeBMPfold::INTER_2;//INTER_2效果最差，莫名其妙
	index=ShapeBMPfold::FOLD_MAP;		//	
	//index=ShapeBMPfold::SIMILAR_TRANS;	//竟然与FOLD_MAP没啥区别，似乎leaf更纯一些，但为何收敛更慢？
	WeakLearner::minSet=MAX(2,Trains.size()/1000);
	nPickWeak = 100;//有意思
	if( regular == MULTI_TREE ){		//multi_tree改变了很多
		eta = 1;		//需要adaptive learning rate,参见multi_1_23_2016.dat
		//maxDepth=5;	//其dat文件较4增加一倍
		//lenda=3.0;
	}else{
	}

	int nTrain=Trains.size( ),nTest=Tests.size(),nz=0;		
	InitFeat( 0x0 );	
	nClass=1;
	hTrainData=new FeatData( nullptr,nTrain,nCand,nClass,0 );		
	
	cand_dis=new float[nCand*nCand]( );
	RandCandidate( nCand,spMean,cands,seed );
#ifdef  _BIT_FEAT_TO_SAVE_SPACE_
	printf( "\n********* BIT FEAT TO SAVE SPACE  *********" );
#endif
	printf( "\n********* %s INDEX: %s nBlitThread=%d *********",regular==MULTI_TREE?"MULTI":"SINGLE",
		ShapeBMPfold::index_info(index).c_str(),nBlitThread );
	//spMean.PtsOnFold( cands,spMean.vY );
	if( nBlitThread>1 ){
		arrPTF=new ShapeBMPfold::PTFs[nBlitThread];
		for( int i=0; i<nBlitThread;i++ )	{
			arrPTF[i].resize( nCand );
			std::copy( cands.begin(),cands.end(),arrPTF[i].begin() );
		}
	}
	//for( int i=0;i<nTest;i++ )	oob_test.push_back( new F4NO(-1.0,i) );
}

void RF_ShapeRegress::DumpTree( int xno,DecisionTree *hTree,int flag ){
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
RF_ShapeRegress::~RF_ShapeRegress(){	
	if( fpC!=NULL )	fclose(fpC);		
	if( fpD!=NULL )	fclose(fpD);	
	if( fpT!=NULL )	fclose(fpT);		
	featLines.clear( );
	if( cand_dis!=nullptr )
		delete[] cand_dis;
	if( arrPTF!=nullptr )	{
		for( int i=0; i<nBlitThread;i++ )	
			arrPTF[i].clear( );
		delete[] arrPTF;
	};
}

double RF_ShapeRegress::ErrorAt( arrPFNO& samps ){
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
void RF_ShapeRegress::Confi_Impuri(  WeakLearner *hWeak,int flag ){
	RandRegress *hRegress=dynamic_cast<RandRegress *>(hWeak);
	GST_VERIFY( hRegress!=nullptr,"RF_ShapeRegress::Impurity" );
	int no,nSamp=hWeak->samps.size();
	//ShapeBMPfold::VECT errV,meanV;
	mSum.setZero( );		hWeak->confi=0.0;
	double impuri=0.0,a1=0,a2=0;
	for each ( F4NO *fn in hWeak->samps  ){
		no=fn->pos;			assert(no>=0 && no<hTrainData->ldF );
		mSum += Trains[no]->vR;		a1+=Trains[no]->vR.squaredNorm();	
		hWeak->confi+=1.0-Trains[no]->NormalErr( );
	}
	hWeak->confi/=nSamp;
	assert( hRegress->sum.size()==0 );
	hRegress->sum=mSum;
	mSum/=nSamp;		a2=nSamp*mSum.squaredNorm();	
	if( !isCalcErr )	{	
		hWeak->impuri=DBL_MAX;
		hWeak->err=DBL_MAX;		return ;		
	}
	hWeak->err=0.0;		
	for each ( F4NO *fn in hWeak->samps  ){
		no=fn->pos;				
		//Trains[no]->Err2Y( mOff );		mOff -= mSum;		
		mOff.noalias() = Trains[no]->vR-mSum;		
		impuri+=mOff.squaredNorm();		
		hWeak->err+=Trains[no]->NormalErr( )/nSamp;	//InterocErr(Trains[no])/nSamp;
	}
	assert( fabs(a1)<FLT_EPSILON || fabs(a1-impuri-a2)<FLT_EPSILON*100*a1 );

	hWeak->impuri=impuri/nSamp;
	return ;
}
bool RF_ShapeRegress::GetFeatDistri( WeakLearner *hWeak,float *distri,int flag ){
	BLIT_Diff *hBlit=dynamic_cast<BLIT_Diff*>(hWeak->hBlit);
	GST_VERIFY( hBlit!=nullptr,"RF_ShapeRegress::Split is 0" );
	float thrsh=hBlit->thrsh,*fu=hTrainData->Feat(hBlit->id),*fv=hTrainData->Feat(hBlit->other);
	int no,i=0,nTrain=Trains.size( )-nOOB;
	for each ( F4NO *fn in hWeak->samps  ){
		no=fn->pos;			assert(no>=0 && no<hTrainData->ldF );		
		if( stage==RF_TEST ){
			GST_VERIFY( no>=nTrain,"RF_ShapeRegress::OOB is X" );
		}else{
			assert( no<nTrain );
		}
		fn->val = fu[no]-fv[no];
//		fn->val = fu[no];
		if( distri!=nullptr )
			distri[i++]=fn->val;
	}
	return true;
}

bool RF_ShapeRegress::LeafModel(  WeakLearner *hWeak,int flag ){
	int no,nSamp=hWeak->samps.size( );
	RandRegress *hRegress=dynamic_cast<RandRegress *>(hWeak);
	GST_VERIFY( hRegress!=nullptr,"" );
	ShapeBMPfold *hX=nullptr;
	if( stage == RF_TRAIN ){
		//if( skdu.isLastTree( ) )			Confi_Regress( hWeak,flag );
		mSum.setZero( );
		for each ( F4NO *fn in hWeak->samps   ){
			no=fn->pos;			assert(no>=0 && no<hTrainData->ldF );
			//Trains[no]->Err2Y( mOff );			mSum += mOff;	
			mSum += Trains[no]->vR;
		}
		hRegress->move = eta*mSum/nSamp;
		int nRow=hRegress->move.rows( ),nCol=hRegress->move.cols( ),r,c;
#ifdef _BIT_FEAT_TO_SAVE_SPACE_
		for( r=0;r<nRow;r++ ){
		for( c=0;c<nCol;c++ ){
			float a = hRegress->move(r,c),b=a<_BIT_FEAT_a0_?_BIT_FEAT_a0_ : a>_BIT_FEAT_a1_ ? _BIT_FEAT_a1_ : a;
			char ch = b>0 ?  _BIT_FEAT_MAX_*b+0.5 : _BIT_FEAT_MAX_*b-0.5;		
			assert( ch>=-127 && ch<=127 );
			hRegress->move(r,c)=ch;
		}
		}
#endif
	}else{
	}

	for each ( F4NO *fn in hWeak->samps   ){
		no=fn->pos;				hX=Trains[no];
		if( regular==SINGLE_TREE )	{
			hX->Move( hRegress->move );
			hX->UpdateErr( );	
		}else{
#ifdef _BIT_FEAT_TO_SAVE_SPACE_
			hX->vMomen += hRegress->move/_BIT_FEAT_MAX_;
#else
			hX->vMomen += hRegress->move;
#endif
			hX->confi += hRegress->confi;
			hX->nBag++;
		}
		
	}
	return true;
}

/*
hBLIT RF_ShapeRegress::GetBlit( WeakLearner *hWeak,int flag ){
	if( hWeak->hBlit==nullptr ){
		hWeak->hBlit=new BLIT_Diff( );
	}
	RandRegress *hRegress=dynamic_cast<RandRegress *>(hWeak);
	GST_VERIFY( hRegress!=nullptr,"RF_ShapeRegress::GetBlit" );
	BLIT_Diff *hBlit=dynamic_cast<BLIT_Diff*>(hWeak->hBlit);
	int i,no,j,nSamp=hWeak->samps.size(),nLeft=0,nRight=nSamp;
	int nPt=hTrainData->nFeat,iU=UniformInt(0,nPt-1),iV=iU;
	//while( iV==iU )		iV=UniformInt(0,nPt-1);
	while( iV==iU ){		iV = (int)(arrPDis[iU]( *hRander ));	}
	featLines.push_back( new F4NO(-1,-1,iU,iV) );		nzWeak+=1.0;

	float f,f0,g,*fu=hTrainData->Feat(iU),*fv=hTrainData->Feat(iV);
	arrPFNO distri=hWeak->samps;
	ShapeBMPfold::VECT errL(mOff),errR(hRegress->sum);
	ShapeBMPfold *hX=nullptr;
	errL.setZero(),		//errR.setZero();
	assert( Trains.size()==hTrainData->ldF );
	for each ( F4NO *fn in distri  ){
		no=fn->pos;			assert(no>=0 && no<hTrainData->ldF );
		fn->val = fu[no]-fv[no];
	}
	if( 0 ){
		std::sort( distri.begin(),distri.end(),F4NO::isPSmall );
		for( j=0; j<nSamp;j++ )	{
			f = distri[j]->val;
			if( nLeft<WeakLearner::minSet )	
			{	goto LOOP;	}
			if( nRight<WeakLearner::minSet )
				break;
			if( f==f0 )
			{	goto LOOP;	}
			g = errL.squaredNorm()/nLeft+errR.squaredNorm()/nRight;
			if( g>hBlit->gain )
			{	hBlit->id = iU;		hBlit->other = iV;	hBlit->thrsh=(f+f0)/2;		hBlit->gain=g;		}
		LOOP:
			assert( j==0||f0<=f );		f0 = f;
			no = distri[j]->pos;		
			//Trains[no]->Err2Y( mOff );		errL+=mOff;		errR-=mOff;
			errL.noalias()+=Trains[no]->vR;				errR.noalias()-=Trains[no]->vR;
			nLeft++;		nRight--;			
		}			
	}else{
		std::uniform_real_distribution<float> dist(0,255.0);
		double thrsh=(dist(*hRander)-128)/2.0;
		for( nLeft=0,j=0; j<nSamp;j++ )	{
			if( distri[j]->val<=thrsh ){
				no = distri[j]->pos;	nLeft++;
				errL.noalias()+=Trains[no]->vR;		
			}
		}
		if( nLeft<WeakLearner::minSet || nRight<WeakLearner::minSet )	
		{	return hBlit;	}			
		errR.noalias()-=errL;		nRight=nSamp-nLeft;
		g = errL.squaredNorm()/nLeft+errR.squaredNorm()/nRight;
		if( g>hBlit->gain )
		{	hBlit->id = iU;		hBlit->other = iV;	hBlit->thrsh=thrsh;		hBlit->gain=g;		}		
	}
	
	return hBlit;
}*/

void RF_ShapeRegress::BlitSamps( WeakLearner *hWeak,SAMPs &left,SAMPs &right,int flag )	{
	assert( left.size()==0 && right.size()==0 );
	RandRegress *hRegress=dynamic_cast<RandRegress *>(hWeak);
	GST_VERIFY( hRegress!=nullptr,"RF_ShapeRegress::GetBlit" );
	BLIT_Diff *hBlit=dynamic_cast<BLIT_Diff*>(hWeak->hBlit);
	GetFeatDistri( hWeak );
	for each ( F4NO *fn in hWeak->samps  ){
		//no=fn->pos;			assert(no>=0 && no<ldF );		
		if( fn->val<=hBlit->thrsh ){
			left.push_back(fn);
		}else
			right.push_back(fn);
	}
	return ;
}

hBLIT RF_ShapeRegress::GetBlit( WeakLearner *hWeak,int flag ){
	if( hWeak->hBlit==nullptr ){
		hWeak->hBlit=new BLIT_Diff( );		
	}
	RandRegress *hRegress=dynamic_cast<RandRegress *>(hWeak);
	GST_VERIFY( hRegress!=nullptr,"RF_ShapeRegress::GetBlit" );
	BLIT_Diff *hBlit=dynamic_cast<BLIT_Diff*>(hWeak->hBlit);
	int i,j,nPt=hTrainData->nFeat,nSamp=hWeak->samps.size(),nRand=this->nPickAtSplit( nullptr ),nCand=cands.size();
	int *iUs=new int[nRand](),*iVs=new int[nRand]( ),iU,iV;
	std::uniform_real_distribution<float> uniFloat(0,1);
	float *thrs=new float[nRand]();
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
#pragma omp parallel for num_threads(nBlitThread) private( i ) 
	for( i=0; i<nRand; i++ ){
		ShapeBMPfold::VECT errL(mOff);
		float a;
		int iU=-1,iV=iU,nLeft=0,nRight=nSamp,no,nzeL=errL.size( );	
		iU=iUs[i];			iV=iVs[i];
	/*	do{
			iU=UniformInt(0,nPt-1);		iV=UniformInt(0,nPt-1);
			a = uniFloat(*hRander);
			if( cand_dis[iU*nCand+iV]>a )
				break;
		}while( 1 );*/
		
		float *eL=errL.data(),one=1.0,thrsh=thrs[i];
		float g,*fu=hTrainData->Feat(iU),*fv=hTrainData->Feat(iV);	
		errL.setZero();		
		for each ( F4NO *fn in hWeak->samps  ){
			no=fn->pos;			assert(no>=0 && no<hTrainData->ldF );
			if( fu[no]-fv[no]<=thrsh ){
		//	if( fu[no]<=thrsh ){
				nLeft++;	
				//errL.noalias()+=Trains[no]->vR;		
				AXPY( nzeL,one,Trains[no]->vR.data(),eL );
			}
		}	
		nRight=nSamp-nLeft;
		if( nLeft<WeakLearner::minSet || nRight<WeakLearner::minSet )	
		{	continue;	}			
		ShapeBMPfold::VECT errR=hRegress->sum-errL;		
		g = errL.squaredNorm()/nLeft+errR.squaredNorm()/nRight;
		if( g>hBlit->gain ){
#pragma omp critical
			if( g>hBlit->gain ) {	hBlit->id = iU;		hBlit->other = iV;	hBlit->thrsh=thrsh;		hBlit->gain=g;		}		
		}
	}	
	delete[] iUs;		delete[] iVs;			delete[] thrs;
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

void RF_ShapeRegress::FeatLineBmp( string sPath,int flag ){
	SHAPE_IMAGE si(256,256,3);		si.bpp=24;
	BMP_DRAW bmp( &si );

	int x0,x1,y0,y1,iU,iV;
	spMean.SetROI( RoCLS( 16,16,224,224 ) );
	spMean.PtsOnFold( cands,spMean.vY );
	for each ( F4NO *hfn in featLines ){
		iU=hfn->x_1;			iV=hfn->x_2;
		x0=cands[iU].gc;		y0=cands[iU].gr;
		x1=cands[iV].gc;		y1=cands[iV].gr;
		bmp.Line( x0,x1,y0,y1 );
	};
	bmp.Save( sPath.c_str() );
}
void RF_ShapeRegress::TraceBmp( string sPath,int type,int flag ){
	PICS_N pics;
	SHAPE_IMAGE si(96,96,3);		si.bpp=24;
	string sNailPath="F:\\GiFace\\trace\\all_";
	int i,nTest=Tests.size(),ldPic = 10;
	switch( type ){
	case 1:
		ldPic = 9;
		for each( F4NO *fn in Tests )	fn->val=Trains[fn->pos]->vR.norm( );
		std::sort( Tests.begin( ),Tests.end(),F4NO::isPSmall );
		for( int i=MAX(0,nTest-10);i<nTest;i++ ){
			ShapeBMPfold *hSamp=Trains[Tests[i]->pos];
			hSamp->TraceBmpS( pics.nails,"",si,0x0);
		}
		for( int i=0;i<MIN(nTest,10);i++ ){
			ShapeBMPfold *hSamp=Trains[Tests[i]->pos];
			hSamp->TraceBmpS( pics.nails,"",si,0x0);
		}
		break;
	case 0:
		ldPic = 20;
		for each( F4NO *fn in Tests )	{
			ShapeBMPfold *hSamp=Trains[fn->pos];
			pics.nails.push_back( hSamp->TraceBmp("",si,0x0) );
			if( pics.nails.size()>=1000 )	break;
		}		
		break;
	}
	
	pics.PlotNails( sPath.c_str(),si,ldPic,true );
}	

void RF_ShapeRegress::AfterTrain( FeatData *hData,int flag ){
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

void RF_ShapeRegress::OnMultiTree( int cas,int nMulti,int flag ){
	double oob0=0,err=0,im1=0,a;
	int total=Trains.size(),nBoot=total-nOOB,no=0,nzBag=0,bag0=0;
	eOOB=0;
	for each( ShapeBMPfold *hFold in Trains ){
		if( no>=nBoot )		oob0 += hFold->NormalErr( )/nOOB;
		//hFold->vS-=hFold->vMomen/nMulti;
		//hFold->lastOff=0;
		if( hFold->nBag>0 ){
			hFold->regressor.push_back( hFold->vMomen.norm()/hFold->nBag );
			//hFold->lastOff=hFold->vMomen.norm()/hFold->nBag;
			hFold->vS-=hFold->vMomen/hFold->nBag;		
			hFold->confi/=hFold->nBag;
		}	else
		{	hFold->regressor.push_back(0.0);	bag0++;	}
		hFold->UpdateErr( );
		if( no<nBoot ){		//
			err+=hFold->NormalErr( );		nzBag+=hFold->nBag;
			assert( hFold->nBag<=nMulti );
		}else{				//oob set
			a = hFold->NormalErr( )/nOOB;			eOOB += a;
			assert( hFold->nBag==nMulti );
		}
		hFold->nBag=0;
		no++;
		hFold->vMomen.setZero( );		hFold->confi=0.0;
	}
	err/=nBoot;
//	printf( "%d: err=(%5.3g,%5.3g) eOOB=%g nTree=%d nWeak=%g,eta=%g\n",step,im1,hRoot->err,eOOB,forest.size(),nzWeak/(step+1),eta );
	printf( "%d-%d: MultiTree bag=%g(%d) err=(%5.3g),eOOB=%g->%g eta=%g@%d\n",cas,skdu.step,nzBag*1.0/nBoot,bag0,err,oob0,eOOB,eta,nOOB );	
}

int RF_ShapeRegress::Train( string sTitle,int cas,int flag )	{
	GST_TIC(tick);	
	
	eOOB = 1.0;
	//skdu.step++;		//rand_seed_0=RAND_REINIT;
	int nSample=Trains.size( ),k;
	UpdateFeat( );
	//hTrainData->Reshape( );
	assert( forest.size()==0 );	

	double im0=0,im1=0,a;
	mSum.setZero( );
	for each( ShapeBMPfold *hFold in Trains ){
		hFold->regressor.clear( );
		hFold->UpdateErr( );	
		mSum+=hFold->vR;
		im0+=hFold->vR.squaredNorm( );
	}
	if( im0/nSample<FLT_EPSILON )
	{	eOOB=0;	printf( "\n********* You are LUCKY *********\n");	return 0x0;	}
	if( 0 ){	//均值平移，似乎没有必要
		mSum/=nSample;	a = nSample*mSum.squaredNorm( );
		for each( ShapeBMPfold *hFold in Trains ){
			hFold->vS-=mSum;			
			hFold->UpdateErr( );//vR=hFold->vS-hFold->vY;
			im1+=hFold->vR.squaredNorm( );
		}
		assert( fabs(im0-a-im1)<FLT_EPSILON*im0 );
	}
	printf( "\n********* %s_%d(%d-%d,im=%g),nOOB=%d,Step=%d,nCand=%d,dep1=%d,eta=%g,lenda=%g,weak=%d minSet=%g...\r\n",
		sTitle.c_str(),cas,Trains.size()-nOOB,nOOB,sqrt(im0/nSample),nOOB,nTree,cands.size( ),maxDepth+1,
		eta,lenda,nPickAtSplit(nullptr),WeakLearner::minSet );
//	RF_ConfiRegress confi(this);

	nzWeak=0.0;
	DForest curF;
	if( isToCPP	)		ToCPP( cas,0x0 );
	double eta0=eta/2,eta1=eta*2,et=(eta1-eta0)*2/nTree;
	int EACH=nTree/skdu.nStep,t,noT=0;
	for( skdu.step=0; skdu.step<skdu.nStep; skdu.step++){
		//adaptive eta	略有加速
		//eta=step<nTree/2 ? eta0+et*step : eta1-et*(step-nTree/2);
		isDumpLeaf = regular==SINGLE_TREE && skdu.step==skdu.nStep-1;
		for( t=0;t<EACH; t++,noT++ ){
			skdu.noT=noT;
			isCalcErr=noT%10==0 || noT==nTree-1;
			DecisionTree *hTree=new DecisionTree( this,hTrainData );			hTree->name=to_string(666)+"_"+to_string(noT);	
			WeakLearner* hRoot=hTree->hRoot( );
			im1 = hRoot->impuri;
			forest.push_back( hTree );
			//isDumpLeaf = noT==nTree-1;
			RandomForest::Train( hTrainData );
			TestOOB( hTrainData );
			if( regular==SINGLE_TREE && isCalcErr ){
				printf( "%d: err=(%5.3g,%5.3g) eOOB=%g nTree=%d nWeak=%g,eta=%g\n",skdu.step,im1,hRoot->err,eOOB,forest.size(),nzWeak/(skdu.step+1),eta );
			}
			k=0;
			for each( DecisionTree *hTree in forest )	{		
				if( isToCPP	)	ToCPP( hTree,cas,skdu.step,k++,0x0 );
				delete hTree;	
			}
			forest.clear( );
		}
		if( regular==MULTI_TREE ){
			OnMultiTree( cas,EACH );
			if( skdu.step<skdu.nStep-1 )	UpdateFeat( );
		}else{
			//原则上move后需要UpdateFeat,但随着move越来越小，并不需要。1_19.dat/1_22.dat的对比也显示了这一点
			if( 1 )		
				UpdateFeat( );
		}
		//confi.Train( sTitle,cas,flag );

		//if( j==0 )			FeatLineBmp( "F:\\GiFace\\trace\\featline.bmp",1 );
	}
	AfterTrain( cas,EACH );
	if( isToCPP	)	{	
		fprintf( fpC,"\n}\n"		);
	}
	TraceBmp( "F:\\GiFace\\trace\\all_"+to_string(cas)+".bmp",0 );
	printf( "********* %s time=%g\r\n\n",sTitle.c_str(),GST_TOC(tick) );

//	confi.Train( sTitle,cas,flag );

	return 0x0;
}

bool RF_ShapeRegress::InitCPP( char *pathC,char *pathD,char *pathT,int cas,int flag ){
	if( !isToCPP	)		return false;
	sPre="RFC";
	if( cas==0 ){
		fpC=fopen( pathC,"wt" );		fpD=fopen( pathD,"wb" );	fpT=fopen( pathT,"wb" );
		//fflush( fpC );
	}else{
		fpC=fopen( pathC,"at" );		fpD=fopen( pathD,"ab" );	fpT=fopen( pathT,"ab" );
	}
	GST_VERIFY( fpC!=NULL && fpD!=NULL,"RF_ShapeRegress::InitCPP failed." );
	::setbuf(fpC,NULL);			::setbuf(fpD,NULL);
	int nz=spMean.vY.count( );
	if( cas==0 ){
		printf( "\n*********  Save this aligner to %s-%s *********\r\n",pathC,pathD );

		fprintf( fpC,
			"#include <stdint.h>\n#include <math.h>\n#include \".\\GruST\\util\\GST_datashape.hpp\"\n"
			"typedef uint8_t BIT_8;\n"
			"static int const ldSp=%d,ldNode=%d;\n\n",nz,5 );	
		fprintf( fpC,
#ifdef  _BIT_MOVE_
			"void _TRE_( float*shape,float*feat,signed char *moves,int *nodes,int flag=0x0){\n"	);
#else
			"void _TRE_( float*shape,float*feat,float *moves,int *nodes,int flag=0x0){\n"	);
#endif
		fprintf( fpC,	"int cur=0,i;\nint *nod;\n"
						"while(cur!=-1){\n\tnod=nodes+cur*ldNode;\n"
						"\tif(nod[0]==-1)	break;\n"
						"\tfloat val = feat[(int)(nod[0])]-feat[(int)(nod[1])];\n"
						"\tif( val<=nod[2] )	cur=(int)(nod[3]); else	cur=(int)(nod[4]);\n"
						"};\n"	
				);
		fprintf( fpC,	
#ifdef  _BIT_MOVE_
						"signed char  *move=moves+(int)(nod[1]);\n" 
#else
						"float *move=moves+(int)(nod[1]);\n" 
#endif
						"for(i=0;i<ldSp;i++)	shape[i]-=move[i]/1000.0;\n" 
						"};\n\n"	
			);
		
	}

	return true;
}

static char bmove[500];
void RF_ShapeRegress::ToCPP( DecisionTree *hTree,int cas,int step,int tree,int flag ){
	WeakLearners vNodes;
	hTree->GetNodes( vNodes );
	int nWeak=vNodes.size( ),i,nRow=mOff.rows(),no=0,ret,nLeaf=0;
	sprintf( sLine,"_TRE_%d_%d_%d",cas,step,tree );
	hTree->name =sLine;
	const char *NAM=hTree->name.c_str();
	for each( WeakLearner* hWeak in vNodes ){
		RandRegress *hRegress=dynamic_cast<RandRegress *>(hWeak);
		GST_VERIFY( hRegress!=nullptr,"RF_ShapeRegress::Impurity" );
		hRegress->id=no++;
		if(hRegress->left!=nullptr)		continue;
		float *move=hRegress->move.data();
#ifdef _BIT_MOVE_
		for( i=0;i<nRow*2;i++ ){
			bmove[i]=move[i];		
			assert( bmove[i]>=-127 && bmove[i]<128 );
		}
		fwrite( bmove,sizeof(char),nRow*2,fpD );	nLeaf++;	
#else
		fwrite( move,sizeof(float),nRow*2,fpD );	nLeaf++;	
#endif
	}
	if( 1 ){
		int nodes[5],nNode=vNodes.size( );
		fwrite( &nNode,sizeof(int),1,fpT );
		fwrite( &nLeaf,sizeof(int),1,fpT );
		nLeaf=0;
		for each( WeakLearner* hWeak in vNodes ){
			RandRegress *hRegress=dynamic_cast<RandRegress *>(hWeak);
			BLIT_Diff *hBlit=dynamic_cast<BLIT_Diff *>(hRegress->hBlit);
			if(hBlit==nullptr){
				nodes[0]=-1,nodes[1]=nLeaf*nRow*2,nodes[2]=0,nodes[3]=-1,nodes[4]=-1;	nLeaf++;
			}else{
				nodes[0]=hBlit->id,		nodes[1]=hBlit->other,		nodes[2]=(int)(hBlit->thrsh),
					nodes[3]=hRegress->left->id,			nodes[4]=hRegress->right->id;
			}
			fwrite( nodes,sizeof(int),5,fpT );
		}
	}else{
		nLeaf=0;
		fprintf( fpC,
			//"void %s( float*shape,float*feat,float **moves,int flag=0x0){\n"	
			"float nodes_%d_[%d][5]={\n",step,nWeak	);
		for each( WeakLearner* hWeak in vNodes ){
			RandRegress *hRegress=dynamic_cast<RandRegress *>(hWeak);
	//		GST_VERIFY( hRegress!=nullptr,"RF_ShapeRegress::Impurity" );
	//		hRegress->id=no++;
			BLIT_Diff *hBlit=dynamic_cast<BLIT_Diff *>(hRegress->hBlit);
			if(hBlit==nullptr){
				fprintf( fpC,"{%d,%d,%d,%d,%d},",-1,nLeaf*nRow*2,0,-1,-1	);		nLeaf++;	
			}else{
				fprintf( fpC,"{%d,%d,%d,%d,%d},",hBlit->id,hBlit->other,(int)(hBlit->thrsh),hRegress->left->id,hRegress->right->id	);	//{	hBlit->id = iU;		hBlit->other = iV;	hBlit->thrsh=thrsh;		hBlit->gain=g;		}		
			}
		}
		fprintf( fpC,"\n};\n"	);
		fprintf( fpC,"_TRE_( shape,feat,moves,nodes_%d_);\n",step );
		fprintf( fpC,
			"*moves+=%d;\n"
			//"}\n"
			,nLeaf*nRow*2
			);
	}
	//ret=fflush(fpC);			
	
	vNodes.clear( );
	return;
}

void RF_ShapeRegress::ToCPP( int cas,int flag ){
	int i,nRow=mOff.rows(),no=0,ret,nStep=nTree,nCand=cands.size( ),ldw;
	char NAM[100];
	sprintf( NAM,"_CAS_%d_",cas );	
	fprintf( fpC,
#ifdef _BIT_MOVE_
		"void %s( BIT_8*gray,int hei,int ldW,float*shape,int **nodes,signed char**moves,RoCLS& roi,int flag=0x0){\n"	
#else
		"void %s( BIT_8*gray,int hei,int ldW,float*shape,int **nodes,float**moves,RoCLS& roi,int flag=0x0){\n"	
#endif
		"int i,j,r,c,mark;\n",NAM,nCand	);

	fprintf( fpC,"float fr,fc,sum[ldSp],feat[%d],cands[%d][3]={\n",nCand,nCand );	
	for ( i=0;i<nCand;i++ ){ 
		fprintf( fpC,"%gf,%gf,%d,",cands[i].dc,cands[i].dr,cands[i].id );
	}
	fprintf( fpC,"\n};\n"		);
	if( 1 ){	//multiple tree
		fprintf( fpC,"for( i = 0; i<%d; i++)	{\n",nTree/10 );
		fprintf( fpC,"for( j=0;j<%d;j++){\n\tmark=(int)(cands[j][2]);\n"
			"\tfc=shape[mark]+cands[j][0],fr=shape[mark+ldSp/2]+cands[j][1];\n"
			"\troi.Map1_f(fc,fr,c,r);\n"
			"\tif( r<0||r>=hei || c<0||c>=ldW)	{	feat[j]=0;	continue;	}\n"
			"\tfeat[j]=gray[r*ldW+c];	\n"
			"}\n"
			,nCand );
		fprintf( fpC,//"for( i = 0; i<%d; i++)	{\n"
			"for( j=0;j<ldSp;j++)	sum[j]=0.0;\nfor( j = 0; j<10; j++)	{\n"
			"int *tree=*nodes,nNode=tree[0],nLeaf=tree[1];\n"
			"_TRE_( sum,feat,*moves,tree+2); \n"
			"*moves+=nLeaf*ldSp;		*nodes+=2+nNode*ldNode;"
			"\n}\nfor( j=0;j<ldSp;j++)	shape[j]+=sum[j]/10;"
			"\n}"	);
	}
	else	{	//single tree
		fprintf( fpC,"for( i=0;i<%d;i++){\n\tmark=(int)(cands[i][2]);\n"
			"\tfc=shape[mark]+cands[i][0],fr=shape[mark+ldSp/2]+cands[i][1];\n"
			"\troi.Map1_f(fc,fr,c,r);\n"
			"\tif( r<0||r>=hei || c<0||c>=ldW)	{	feat[i]=0;	continue;	}\n"
			"\tfeat[i]=gray[r*ldW+c];	\n"
			"}\n"
			,nCand );
		fprintf( fpC,"for( i = 0; i<%d; i++)	{\n"
			"int *tree=*nodes,nNode=tree[0],nLeaf=tree[1];\n"
			"_TRE_( shape,feat,*moves,tree+2); \n"
			"*moves+=nLeaf*ldSp;		*nodes+=2+nNode*ldNode;\n"
			"\n}",nTree	);	
	}
	return;
}

void RF_ShapeRegress::CoreInCPP( int nCas,int flag ){
	if( fpC==NULL )		return;
	int i,nRow=mOff.rows(),no=0,ret,nStep=nTree,nCand=cands.size( ),ldw;
	char NAM[100];
	sprintf( NAM,"Face_Align_" );	
	fprintf( fpC,
#ifdef _BIT_MOVE_
		"void %s( BIT_8*gray,int hei,int ldW,float*shape,int **nodes,signed char**moves,RoCLS& roi,int nMost,int flag=0x0){\n"	
#else
		"void %s( BIT_8*gray,int hei,int ldW,float*shape,int **nodes,float**moves,RoCLS& roi,int nMost,int flag=0x0){\n"	
#endif
		"\tint i=0;\n",NAM,nCand	);
	for ( i=0;i<nCas;i++ ){
		fprintf( fpC,"\t_CAS_%d_( gray,hei,ldW,shape,nodes,moves,roi,flag ); "
			"\tif(++i>=nMost)	return;\n"
			,i	);	
	}	
	fprintf( fpC,"}\n"		);
	return;
}