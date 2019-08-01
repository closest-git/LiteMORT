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

	struct Salp {
		double cost;
		vector<double> position;

		Salp(int dim, int flag = 0x0) {
			position.resize(dim);
		}
		virtual void Copy(const Salp*src,int flag=0x0) {
			position = src->position;
			cost = src->cost;
		}

		//aA+b*B
		virtual void MixPosition(double alpha, const Salp*A, double beta, const Salp*B, int flag) {
			int dim = position.size(), i;
			for (i = 0; i < dim; i++) {
				position[i] = alpha*A->position[i] + beta*B->position[i];
			}
		}
	};

	class BinarySwarm {

	protected:
		Salp *bound = nullptr;		//二分类，初始化为1向量
		Salp *food = nullptr,*leader = nullptr;
		typedef enum{
			SIGMOID
		} TRANSFER_FUNC;
		double *velocity=nullptr, *positon = nullptr;
		vector<Salp*>salps;

		virtual void InitBound(int dim, int flag = 0x0);
		virtual void UpdateLeader(double loss, int flag = 0x0) {}

	public:
		BinarySwarm(int nBird_, int flag = 0x0);

		virtual ~BinarySwarm() {
			if (velocity != nullptr)		delete[] velocity;
			if (positon != nullptr)			delete[] positon;
		}

		

		virtual void Step(double loss, int flag = 0x0){}

	};

	class BSA_salp : public BinarySwarm {
	protected:
		double c_1, c_2, c_3;
		virtual void UpdateLeader(double loss, int flag = 0x0);
	public:		
		BSA_salp(int nBird_,int dim,int flag = 0x0);
		virtual void Step(double loss, int flag = 0x0);
	};


}