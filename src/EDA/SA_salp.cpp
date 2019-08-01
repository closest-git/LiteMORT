
#include "SA_salp.hpp"
using namespace Grusoft;

//特殊性，初始化为1向量
void BinarySwarm::InitBound(int dim,int flag) {

}

BinarySwarm::BinarySwarm(int nBird_, int flag) {

}

BSA_salp::BSA_salp(int nBird_, int dim, int flag) :BinarySwarm(nBird_,flag) {
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

void BSA_salp::Step(double loss, int flag) {
	UpdateLeader(loss,flag);
	//UpdatePosition
	double cost = salps[0]->cost;

	Salp *pre = nullptr;
	for (auto salp : salps) {
		if (salp == leader)
			continue;
		salp->MixPosition(0.5, salp, 0.5, pre, 0x0);
	}

	//update food
	Salp *salp_0 = nullptr;
	for( auto salp : salps ){
		if (cost > salp->cost) {
			cost = salp->cost;		salp_0 = salp;
		}
	}
	food->Copy(salp_0);

	//food.set
	return;
}