#pragma once
#include <vector>
#include <map>  
#include <algorithm>
#include <cmath>
#include "../util/samp_set.hpp"

typedef double BINFOLD_FLOAT;

namespace Grusoft {
	class FeatsOnFold;
	class FRUIT;

	struct BinSalp {
		double cost;
		vector<double> position;

		BinSalp(int dim, int flag = 0x0);
		BinSalp(int dim, const vector<int>&picks, int flag = 0x0);
		BinSalp(const vector<bool>&pick_mask, int flag = 0x0);		//类似于boost中的 integer_mask

		virtual void Copy(const BinSalp*src,int flag=0x0) {
			position = src->position;
			cost = src->cost;
		}

		//aA+b*B
		virtual void MixPosition(double alpha, const BinSalp*A, double beta, const BinSalp*B, int flag) {
			int dim = position.size(), i;
			for (i = 0; i < dim; i++) {
				position[i] = alpha*A->position[i] + beta*B->position[i];
			}
		}
	};

	//special swarm algortim on GBDT trees
	class BinarySwarm_GBDT {

	protected:
		bool first_step = true;
		BinSalp *bound = nullptr;		//二分类，初始化为1向量
		BinSalp *food = nullptr,*leader = nullptr;
		typedef enum{
			SIGMOID
		} TRANSFER_FUNC;
		double *velocity=nullptr, *positon = nullptr;
		vector<BinSalp*>salps;

		virtual void InitBound(int dim, int flag = 0x0);
		virtual void UpdateFood(int flag = 0x0);
		virtual void UpdateLeader(double loss, int flag = 0x0) {}
		virtual void CreateSwarms(int nSalp, int flag = 0x0);
	public:
		BinarySwarm_GBDT(int nMostBird_, int flag = 0x0);

		virtual ~BinarySwarm_GBDT() {
			if (velocity != nullptr)		delete[] velocity;
			if (positon != nullptr)			delete[] positon;
		}

		virtual void AddSalp(int dim, const vector<int>&picks, int flag=0x0) {			
			BinSalp *salp = new BinSalp(dim,picks,flag);
			salps.push_back(salp);
		}
		
		virtual void SetCost(double cost, int flag = 0x0) {
			assert(salps.size()>0);
			BinSalp *salp = salps[salps.size() - 1];
			salp->cost = cost;
		}

		virtual bool Step(int nSalp,int flag = 0x0){
			return false;
		}

		virtual bool GetPicks(vector<int>&picks, int flag = 0x0) {
			return false;
		}

	};

	class BSA_salp : public BinarySwarm_GBDT {
	protected:
		double c_1, c_2, c_3;
		virtual void UpdateLeader(double loss, int flag = 0x0);
	public:		
		BSA_salp(int nBird_,int dim,int flag = 0x0);
		virtual bool Step(int nSalp, int flag = 0x0);
	};


}