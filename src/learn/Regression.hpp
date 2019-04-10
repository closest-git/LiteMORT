#pragma once

namespace Grusoft {
	class Regression {
		double _slope, _yInt;
	public:
		Regression(string alg,int flag=0x0) {

		}

		/*
			https://web.archive.org/web/20150715022401/http://faculty.cs.niu.edu/~hutchins/csci230/best-fit.htm
			Y = Slope * X + YInt
		*/
		template<typename Tx, typename Ty>
		bool Fit(size_t nSamp, tpSAMP_ID *samps,Tx *arrX, Ty *arrY, int flag = 0x0) {
			assert (nSamp >= 2) ;
			double sumX = 0, sumY = 0, sumXY = 0, sumX2 = 0;
			Tx x,y;
			tpSAMP_ID samp;
			for (int i = 0; i<nSamp; i++) {
				samp = samps[i];
				x = arrX[samp], y = arrY[samp];
				sumX += x;				sumY += y;
				sumXY += x*y;			sumX2 += x*x;
			}
			double xMean = sumX / nSamp;
			double yMean = sumY / nSamp;
			double denominator = sumX2 - sumX * xMean;
			// You can tune the eps (1e-7) below for your specific task
			if (std::fabs(denominator) < 1e-7) {
				// Fail: it seems a vertical line
				return false;
			}
			_slope = (sumXY - sumX * yMean) / denominator;
			_yInt = yMean - _slope * xMean;
			return true;
		}

		template<typename Tx>
		Tx At(Tx x, int flag = 0x0) {
			double y = _slope*x+ _yInt;
			return (Tx)(y);
		}
	};

	
}