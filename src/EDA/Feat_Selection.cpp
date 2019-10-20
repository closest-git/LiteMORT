#include "Feat_Selection.hpp"
using namespace Grusoft;

GRander LogicSalp::rander_(2020);
GRander Feature_Selection::rander_(2020);

LogicSalp::LogicSalp(const int dim, int flag) {
	position.resize(dim);
}

LogicSalp::LogicSalp(const int dim, const vector<int>&picks, int flag) {
	position.resize(dim);
	for (int i = 0; i < dim; i++) {
		position[i] = 0;
	}
	for (auto x : picks) {
		position[x] = 1;
	}
}

void Feature_Selection::GetPicks(const LogicSalp *salp, vector<int>&picks, bool isMask, int flag ) {
	int i;
	if (isMask) {
		picks.resize(DIM_);
		for (i = 0; i < DIM_; i++) {
			picks[i] = salp->position[i] == 1;
		}
	}
	else {
		picks.clear();
		for (i = 0; i < DIM_; i++) {
			if (salp->position[i] == 1)
				picks.push_back(i);
			else
				assert(salp->position[i] == 0);
		}

	}
}

void LogicSalp::cross_over(const LogicSalp*A, const LogicSalp*B, int flag) {
	assert(space == BIT_MASK);
	int DIM = position.size(),i;
	int pos = rander_.RandInt32() % DIM;
	for (i = 0; i < DIM; i++) {
		position[i] = i < pos ? A->position[i] : B->position[i];
	}


}
void LogicSalp::mutatioin(double T_mut, int flag) {
	assert(space == BIT_MASK);
	int DIM = position.size(),i;
	for (i = 0; i < DIM; i++) {
		double p = rander_.Uniform_(0, 1);
		if (p < T_mut) {
			position[i] = position[i] == 1 ? 0 : 1;
		}	
	}

}

Feature_Selection::Feature_Selection(int nMostSalp_, int dim_, int flag) : nMostSalp(nMostSalp_), DIM_(dim_),cand(dim_) {
	T_mute = 1.0 / DIM_;		//mutate one feature of each individual (statistically).
	T_elitism = min(T_elitism, nMostSalp_ / 2-1);
}


FS_gene_::FS_gene_(const string nam_, int nSalp_, int dim_, int nMaxIter_, int flag) : Feature_Selection(nSalp_, dim_, flag), offspring(dim_){
	//InitBound(dim);

	printf("\nFS_gene_(\"%s\",nSalp_=%d,dim=%d)", nam_.c_str(),nSalp_, dim_);

}

vector<int> Feature_Selection::roulette_sample(int nPick, vector<float> roulette,int flag) {
	int no = -1,i,j;
	double sum=0;
	vector<float> p_sum;
	vector<int> picks;
	for (i = 0; i < roulette.size(); i++) {
		sum += roulette[i];
		p_sum.push_back(sum);
	}
	for (i = 0; i < nPick; i++) {
		double a = rander_.Uniform_(0, sum);
		for (j = 0; j < roulette.size(); j++) {
			if (a < p_sum[j]) {
				picks.push_back(j);
				break;
			}
		}
	}
	return picks;
}

//remainder stochastic sampling
void FS_gene_::Intermediate_Select(int flag) {
	inter_samps_.clear();
	int i,nPick= salps.size()/2-T_elitism,nSalp= salps.size();
	vector<tpSAMP_ID> idx;
	vector<float> val_c,roulette;
	for (i = 0; i < salps.size(); i++) {
		idx.push_back(i);
		val_c.push_back(salps[i]->fitness);
	}
	std::sort(idx.begin(), idx.end(), [&val_c](tpSAMP_ID i1, tpSAMP_ID i2) {return val_c[i1] > val_c[i2]; });

	for (i = 0; i < T_elitism; i++) {
		inter_samps_.push_back(salps[idx[i]]);
	}

	for (i = T_elitism; i < nSalp; i++) {
		LogicSalp *salp = salps[idx[i]];
		//roulette.push_back(salp->fitness);
		roulette.push_back(nSalp-i);		//Rank based
	}
	vector<int> picks = roulette_sample(nPick,roulette);
	for (auto pick : picks) {
		inter_samps_.push_back(salps[idx[pick]]);
	}

}

//比较有意思，只生成一条
bool FS_gene_::PickOnStep(int nSalp, vector<int>&picks, bool isMask, int flag) {
	if (iter == 0) {
		//printf("\nFS_gene_::PickOnStep...iter=%d......", iter);
	}
	Intermediate_Select();
	vector<int> pick_2 = rander_.kSampleInN(2, inter_samps_.size());
	offspring.cross_over(inter_samps_[pick_2[0]], inter_samps_[pick_2[1]]);
	offspring.mutatioin(T_mute);
	GetPicks(&offspring,picks, isMask);
	iter = iter + 1;
	return true;
}

void Feature_Selection::AddCandSalp(int flag) {
	if (salps.size() >= nMostSalp) {
		LogicSalp *sp = salps[0];
		salps.erase(salps.begin());
		delete sp;
	}
	int dim = cand.DIM();
	LogicSalp *salp = new LogicSalp(dim);
	salp->Copy(&cand);
	salps.push_back(salp);
}

void Feature_Selection::AddSalp(int dim, const vector<int>&picks, int x_, int flag) {
	if (salps.size() >= nMostSalp) {
		LogicSalp *sp = salps[0];
		salps.erase(salps.begin());
		delete sp;
	}
	LogicSalp *salp = new LogicSalp(dim, picks, flag);
	salp->x = x_;
	salps.push_back(salp);
}

