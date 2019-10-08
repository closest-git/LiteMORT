
#include "SA_salp.hpp"
#include <math.h>
using namespace Grusoft;

GRander BinSalp::rander_(2019);
GRander BinarySwarm_GBDT::rander_(2020);

BinSalp::BinSalp(int dim, int flag) {
	position.resize(dim);
}
BinSalp::BinSalp(int dim, const vector<int>&picks, int flag) {
	position.resize(dim);
	for (int i = 0; i < dim; i++) {
		position[i] = rander_.Uniform_(0,0.4);
	}
	for (auto x : picks) {
		position[x] = rander_.Uniform_(0.6,1);
	}
}
BinSalp::BinSalp(const vector<bool>&pick_mask, int flag) {
	int dim = pick_mask.size();
	position.resize(dim);
	for (int i = 0; i < dim; i++) {
		position[i] = pick_mask[i] ? rander_.Uniform_(0, 0.4) : rander_.Uniform_(0.6, 1);
	}
}

//特殊性，初始化为1向量
void BinarySwarm_GBDT::InitBound(int dim,int flag) {

}

BinarySwarm_GBDT::BinarySwarm_GBDT(int nMostSalp_,int dim_,int flag) : nMostSalp(nMostSalp_),DIM_(dim_){

}

BSA_salp::BSA_salp(int nBird_, int dim_, int nMaxIter_, int flag) : BinarySwarm_GBDT(nBird_, dim_,flag) {
	c_1 = 0;
	//InitBound(dim);

}

//http://www.alimirjalili.com/SSA.html
void BSA_salp::UpdateLeader(double loss, int flag) {
	UpdateC1();
	int i;
	double a, b;
	for (i = 0; i < DIM_; i++) {
		c_2 = rander_.Uniform_(0, 1);
		c_3 = rander_.Uniform_(0, 1);
		double a = food->position[i];
		b = (c_3 < 0.5) ? a- c_1*c_2 : a+ c_1*c_2;
		leader->position[i] = b < 0 ? 0 : b>1 ? 1 : b;
	}
	//leader->MixPosition(1, food, c_1*c_2, bound, 0x0);
	return;
}
void BSA_salp::UpdateC1( int flag) {
	double a = 4.0*iter / maxIter;
	c_1 = 2.0*exp(-a*a);
}

void BinarySwarm_GBDT::NormalSwarms(int nSalp, int flag) {
	assert(salps.size() >= nSalp);
	//salps.resize(nSalp);
	std::sort(salps.begin(), salps.end());

	food = new BinSalp(DIM(), flag);
	//UpdateFood();
	//food->Copy(salp_0);
}

void BinarySwarm_GBDT::UpdateFood(int flag) {
	//update food
	BinSalp *salp_0 = nullptr;
	double cost = salps[0]->cost;
	for (auto salp : salps) {
		if (cost > salp->cost) {
			cost = salp->cost;		salp_0 = salp;
		}
	}
	food->Copy(salp_0);
}

bool BSA_salp::Step(int nSalp, int flag) {
	assert(salps.size() >= nSalp);
	if (iter==0) {
		NormalSwarms(nSalp);
	}
	UpdateLeader(0,flag);
	//UpdatePosition

	BinSalp *pre = nullptr;
	for (auto salp : salps) {
		if (salp != leader)
			salp->MixPosition(0.5, salp, 0.5, pre, 0x0);
		pre = salp;
	}
	UpdateFood();
	iter = iter + 1;
	//food.set
	return true;
}


BSA_gene::BSA_gene(int nBird_, int dim_, int nMaxIter_, int flag) : BinarySwarm_GBDT(nBird_, dim_, flag) {


}
bool BSA_gene::Step(int nSalp, int flag) {
	assert(salps.size() >= nSalp);
	if (iter == 0) {
		NormalSwarms(nSalp);
	}
	UpdateLeader(0, flag);
	//UpdatePosition

	BinSalp *pre = nullptr;
	for (auto salp : salps) {
		if (salp != leader)
			salp->MixPosition(0.5, salp, 0.5, pre, 0x0);
		pre = salp;
	}
	UpdateFood();
	iter = iter + 1;
	//food.set
	return true;
}