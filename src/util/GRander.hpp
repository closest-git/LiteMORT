#pragma once
#include <string>
#include <random>
#include <stdint.h>
#include <vector>
#include <set>

#define rotl(r,n) (((r)<<(n)) | ((r)>>((8*sizeof(r))-(n))))

/*
	需要改进，DIST_Normal,DIST_RangeN设计较好
	http://www.drdobbs.com/tools/fast-high-quality-parallel-random-number/229625477?pgno=2
*/
namespace Grusoft{
	class GRander {
		unsigned int x = 123456789;		//过于简单，有问题
		uint64_t xx, yy, zz;
		uint64_t RandRersResrResdra() {  // Combined period = 2^116.23
			xx = rotl(xx, 8) - rotl(xx, 29);                 //RERS,   period = 4758085248529 (prime)
			yy = rotl(yy, 21) - yy;  yy = rotl(yy, 20);      //RESR,   period = 3841428396121 (prime)
			zz = rotl(zz, 42) - zz;  zz = zz + rotl(zz, 14); //RESDRA, period = 5345004409 (prime)
			return xx ^ yy ^ zz;
		}

		inline int RandInt16()		{
			//x = (214013 * x + 2531011);
			x = RandRersResrResdra();
			return static_cast<int>((x >> 16) & 0x7FFF);
		}
		inline float NextFloat()	{
			return static_cast<float>(RandInt16()) / (32768.0f);
		}

	protected:
		//std::ranlux64_base_01 eng;
		std::random_device device;
		uint32_t seed;

	public:
		GRander() {
			std::random_device rd;
			auto genrator = std::mt19937(rd());
			std::uniform_int_distribution<int> distribution(0, x);
			x = distribution(genrator);
		}
		GRander(uint32_t seed_) { Init(seed_);	 }

		virtual void Init(uint32_t seed_) {
			seed = seed_;	 

			unsigned n;
			xx = 914489ULL;
			yy = 8675416ULL;
			zz = 439754684ULL;
			for (n = ((seed >> 22) & 0x3ff) + 20; n>0; n--) { xx = rotl(xx, 8) - rotl(xx, 29); }
			for (n = ((seed >> 11) & 0x7ff) + 20; n>0; n--) { yy = rotl(yy, 21) - yy;  yy = rotl(yy, 20); }
			for (n = ((seed) & 0x7ff) + 20; n>0; n--) { zz = rotl(zz, 42) - zz;  zz = rotl(zz, 14) + zz; }
			
			x = seed;

		}
		inline int RandInt32() {
			//x = (214013 * x + 2531011);
			x = RandRersResrResdra();
			return static_cast<int>(x & 0x7FFFFFFF);
		}
		inline double Uniform_(double a0,double a1) {
			int cur = RandInt32();
			double a = cur*1.0 / 0x7FFFFFFF;
			assert(a>=-1.0 && a <=1.0);
			double b = a0 + (a1 - a0)*(a + 1) / 2.0;
			return b;
		}

		/*virtual size_t operator()(size_t n)	{
			std::uniform_int_distribution<size_t> d(0, n ? n - 1 : 0);
			return d(g);
		}*/

		/*
			K sample in N	(K<=N)
			v0.1	为了和lightgbm对比
				3/2/2019
		*/
		inline std::vector<int> kSampleInN(int K,int N, bool isOrder=true,int flag=0x0) {
			std::vector<int> ret;
			ret.reserve(K);
			if (K > N || K <= 0) {
				return ret;
			}	else if (K == N) {
				for (int i = 0; i < N; ++i) {
					ret.push_back(i);
				}
			}
			else if (K > 1 && K > (N / std::log2(K))) {
				for (int i = 0; i < N; ++i) {
					double prob = (K - ret.size()) / static_cast<double>(N - i);
					if (NextFloat() < prob) {
						ret.push_back(i);
					}
				}
			}
			else {
				std::set<int> sample_set;
				while (static_cast<int>(sample_set.size()) < K) {
					int next = RandInt32() % N;
					if (sample_set.count(next) == 0) {
						sample_set.insert(next);
					}
				}
				for (auto iter = sample_set.begin(); iter != sample_set.end(); ++iter) {
					ret.push_back(*iter);
				}
			}
			return ret;
		}
	};

	/*
	//standard normal distribution
	class DIST_Normal : public GRander	{
		std::normal_distribution<> d;
	public:
		DIST_Normal(int seed) : GRander(seed) {		}

		template<typename T>
		T gen(){
			double a = d(g);
			return T(a);
		}
	};

	//standard normal distribution in [min-max]
	class DIST_RangeN : public GRander	{
		std::normal_distribution<> d;
		double rMin, rMax;
	public:
		DIST_RangeN(int seed, double a0, double a1);
		double gen();
	};*/
}