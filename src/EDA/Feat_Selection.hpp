#pragma once
#include <vector>
#include <map>  
#include <algorithm>
#include <cmath>
#include "../util/samp_set.hpp"
#include "../util/GRander.hpp"

namespace Grusoft {

	//游荡在特征空间里的海鞘
	struct FeatSalp {
		static GRander rander_;
		typedef enum {
			BIT_MASK
		}SPACE_TYPE;
		SPACE_TYPE space=BIT_MASK;

		int x=0;
		float cost;
		vector<double> position;
		FeatSalp(int dim, int flag = 0x0);
		FeatSalp(int dim, const vector<int>&picks, int flag = 0x0);
		FeatSalp(const vector<bool>&pick_mask, int flag = 0x0);		//类似于boost中的 integer_mask

		virtual void Copy(const FeatSalp*src,int flag=0x0) {
			position = src->position;
			cost = src->cost;
			x = src->x;
		}

		//aA+b*B
		virtual void MixPosition(double alpha, const FeatSalp*A, double beta, const FeatSalp*B, int flag) {
			int dim = position.size(), i;
			for (i = 0; i < dim; i++) {
				position[i] = alpha*A->position[i] + beta*B->position[i];
			}
		}

		virtual void cross_over(const FeatSalp*A,  const FeatSalp*B,int flag=0x0);
		virtual void mutatioin(double T_mut,int flag=0x0);
	};

	//special swarm algortim on GBDT trees
	class Feature_Selection {

	protected:
		static GRander rander_;
		typedef enum {
			SIGMOID
		} TRANSFER_FUNC;

		int DIM, iter = 0, maxIter;		// the current iteration and the maximum number of iterations
		double T_mute = 0.01;
		int nMostSalp = 0;
		vector<FeatSalp*>salps;
		FeatSalp *food = nullptr, *leader = nullptr;

	public:
		Feature_Selection(int nMostSalp_,int dim_, int flag = 0x0);


		virtual ~Feature_Selection() {

		}

		virtual void AddSalp(int dim, const vector<int>&picks,int x_, int flag=0x0) {		

		}
		
		virtual void SetCost(double cost, int flag = 0x0) {

		}

		virtual bool Step(int nSalp,int flag = 0x0){	throw "!!!BinarySwarm_GBDT Step is ...!!!";		}

		virtual bool GetPicks(vector<int>&picks, int flag = 0x0) {
			return false;
		}

	};

	class FS_gene_ : public Feature_Selection {
	protected:
		vector<FeatSalp*> samps_;
		FeatSalp  offspring;
		virtual void UpdateLeader(double loss, int flag = 0x0);
		virtual void Intermediate_Select(int flag=0x0);
		virtual void Muatate(int flag = 0x0);
	public:
		FS_gene_(int nBird_, int dim_, int nMaxIter_,int flag = 0x0);
		virtual bool SubStep_1(int nSalp, int flag = 0x0);
	};


}