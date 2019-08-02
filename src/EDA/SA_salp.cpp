
#include "SA_salp.hpp"
using namespace Grusoft;

BinSalp::BinSalp(int dim, int flag) {
	position.resize(dim);
}
BinSalp::BinSalp(int dim, const vector<int>&picks, int flag) {
	position.resize(dim);
}
BinSalp::BinSalp(const vector<bool>&pick_mask, int flag) {

}

//特殊性，初始化为1向量
void BinarySwarm_GBDT::InitBound(int dim,int flag) {

}

BinarySwarm_GBDT::BinarySwarm_GBDT(int nBird_, int flag) {

}

BSA_salp::BSA_salp(int nBird_, int dim, int flag) : BinarySwarm_GBDT(nBird_,flag) {
	c_1 = 0;
	InitBound(dim);

}

//http://www.alimirjalili.com/SSA.html
void BSA_salp::UpdateLeader(double loss, int flag) {
	c_1 = 2 * exp(1.0);
	if (c_3 < 0.5)		c_1 = -c_1;

	leader->MixPosition(1, food, c_1*c_2, bound, 0x0);
	return;
}

void BinarySwarm_GBDT::CreateSwarms(int nSalp, int flag) {
	assert(salps.size() >= nSalp);
	salps.resize(nSalp);
	UpdateFood();
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
	if (first_step) {
		CreateSwarms(nSalp);
		first_step = false;
	}
	UpdateFood(0);
	UpdateLeader(0,flag);
	//UpdatePosition

	BinSalp *pre = nullptr;
	for (auto salp : salps) {
		if (salp == leader)
			continue;
		salp->MixPosition(0.5, salp, 0.5, pre, 0x0);
	}
	UpdateFood();

	//food.set
	return true;
}