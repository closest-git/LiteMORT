#include "Feat_Selection.hpp"
using namespace Grusoft;

GRander FeatSalp::rander_(2020);
GRander Feature_Selection::rander_(2020);

void FeatSalp::cross_over(const FeatSalp*A, const FeatSalp*B, int flag) {
	assert(space == BIT_MASK);
	int DIM = position.size(),i;
	int pos = rander_.RandInt32() % DIM;
	for (i = 0; i < DIM; i++) {
		position[i] = i < pos ? A->position[i] : B->position[i];
	}


}
void FeatSalp::mutatioin(double T_mut, int flag) {
	assert(space == BIT_MASK);
	int DIM = position.size();
	double p = rander_.Uniform_(0, 1);
	if (p < T_mut) {
		int pos = rander_.RandInt32() % DIM;
		position[pos] = position[pos] == 1 ? 0 : 1;
	}
}

Feature_Selection::Feature_Selection(int nMostSalp_, int dim_, int flag) : nMostSalp(nMostSalp_), DIM(dim_) {
	T_mute = 1.0 / DIM;		//mutate one feature of each individual (statistically).
}

//remainder stochastic sampling
void FS_gene_::Intermediate_Select(int flag) {
	
}

//比较有意思，只生成一条
bool FS_gene_::SubStep_1(int nSalp, int flag) {
	Intermediate_Select();
	vector<int> pick_2 = rander_.kSampleInN(2, samps_.size());
	offspring.cross_over(samps_[pick_2[0]], samps_[pick_2[1]]);
	offspring.mutatioin(T_mute);
	return true;
}