#pragma once

#include <memory>
#include <string>
#include <vector>
#include <algorithm>
#include <map>
#include <numeric>
#include <time.h>
using namespace std;
#include "./FeatVector.hpp"

#ifdef WIN32
#include <tchar.h>
#include <assert.h>
#else    
#include <assert.h>
//#define assert(cond)
#endif


namespace Grusoft {
	class MT_BiSplit;

	class FeatsOnFold;
	class Distribution;

	struct FeatPresent {
		FeatVector *hFeat = nullptr;
		float T_min = 5;
		FeatPresent(FeatVector *hF, float T_,int flag=0x0) : hFeat(hF),T_min(T_)	{

		}
		
	};

	class Representive {
		vector<FeatPresent*> arrPFeat;
	public:
		Representive() {

		}
		virtual ~Representive() {
			for (auto pf : arrPFeat)
				delete pf;
			arrPFeat.clear();
		}
		void Append(FeatVector *hF, float T_, int flag = 0x0) {
			arrPFeat.push_back(new FeatPresent(hF, T_));
		}
		bool isValid(const MT_BiSplit *hNode,int flag=0x0);
		void dump(int flag=0x0);
	};


}

