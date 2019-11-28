#include "../data_fold/Loss.hpp"
using namespace Grusoft;

//LambdaRank FeatVec_LOSS::HT_lambda_;

/*
	LightGBM
		$4.7 Applications and Metrics
		default={l2 for regression}, {binary_logloss for binary classification},{ndcg for lambdarank},
		$6.6 "metric to be evaluated on the evaluation sets in addition to what is provided in the training argument"
		type=multi-enum, options=l1,l2,ndcg,auc,binary_logloss,binary_error...

*/
void FeatVec_LOSS::Clear() {
	down.clear();				sample_down.clear();
	resi.clear();				delta_step.clear();
	hessian.clear();			sample_hessian.clear();

	if (samp_weight != nullptr)			delete[] samp_weight;
	if (y != nullptr) {
		if (y->hDistri != nullptr)
			delete y->hDistri;
		delete y;
	}
	if (predict != nullptr)		delete predict;
}
/*
	目前仅UpdateResi
*/
void FeatVec_LOSS::Update(FeatsOnFold *hData_,int round, int flag) {
	if (tpResi == is_float)
		UpdateResi<float>(hData_, round,flag);
	else if (tpResi == is_double)
		UpdateResi<double>(hData_, round, flag);
	else
		throw "FeatVec_LOSS::UpdateResi type is !!!";
	return;
}

void FeatVec_LOSS::InitSampWeight(int flag) {
	bool isTrain = BIT_TEST(flag, FeatsOnFold::DF_TRAIN);
	bool isEval = BIT_TEST(flag, FeatsOnFold::DF_EVAL);
	bool isPredict = BIT_TEST(flag, FeatsOnFold::DF_PREDIC);
	size_t nSamp = size();
	if (isTrain) {
		samp_weight = new float[nSamp]();
		for (size_t i = 0; i < nSamp; i++) {
			samp_weight[i] = 1;	// Y_[i] == 0 ? 1 : 10;
		}
	}
}

string FeatsOnFold::LOSSY_INFO(double err, int flag) {
	char tmp[1000];
	if (config.eval_metric == "auc") {
		sprintf(tmp, "%-8.5g", 1-err);
	}
	else {
		sprintf(tmp, "%-8.5g", err);
	}
	return tmp;
}

double FeatVec_LOSS::ERR(int flag) {
	double err = 0;
	if (hBaseData_->config.eval_metric == "mse") {
		err = hBaseData_->lossy->err_rmse;
		err = err*err;
	}	else if (hBaseData_->config.eval_metric == "rmse") {
		err = hBaseData_->lossy->err_rmse;
	}	else if (hBaseData_->config.eval_metric == "mae") {
		err = hBaseData_->lossy->err_mae;
	}
	else if (hBaseData_->config.eval_metric == "logloss") {
		err = hBaseData_->lossy->err_logloss;
	}
	else if (hBaseData_->config.eval_metric == "auc") {	//需要与EARLY_STOPPING::isOK统一
		err = 1 - hBaseData_->lossy->err_auc;
		//err = hData_->lossy->err_auc;
	}
	return err;
}

bool FeatVec_LOSS::isOK(int typ, double thrsh, int flag) {
	double err = ERR(flag);
	if (hBaseData_->config.eval_metric == "auc") {
		return err < thrsh;
	}else
		return err < thrsh;
	//return err_rmse < thrsh;
}

void FeatVec_LOSS::EDA( ExploreDA *edaX, int flag) {
	const LiteBOM_Config&config = hBaseData_->config;
	size_t nSamp = hBaseData_->nSample();
	bool isPredict = BIT_TEST(flag, FeatsOnFold::DF_PREDIC);
	bool isEval = BIT_TEST(flag, FeatsOnFold::DF_EVAL);
	bool isTrain = BIT_TEST(flag, FeatsOnFold::DF_TRAIN);
	if(config.verbose>0)
		printf("********* FeatVec_LOSS::EDA@\"%s\"\tsamp_weight=%p...\n",hBaseData_->nam.c_str(),samp_weight);
	if (isPredict) {

	}	else {
		//y->EDA(config,true, nullptr, 0x0);
		y->InitDistri(nullptr, true, nullptr, 0x0);
		if(y->hDistri!=nullptr)
			y->hDistri->Dump(-1, false, flag);
		size_t dim = size(),i,nOutlier;
		if (config.objective == "outlier") {
			y->loc(outliers,1);
			nOutlier = outliers.size();
			if (nOutlier == 0 && isTrain)
				throw "FeatVec_LOSS::EDA outlier is 0!!! Please check the value of Y!!!";
			if (nOutlier == 0) {
				printf("\toutliers=%lld!!!\n", nOutlier);
			}	else {
				printf("\toutliers=%lld(%d,...%d)\n", nOutlier, outliers[0], outliers[nOutlier-1]);
			}
		}else if (hBaseData_->config.objective == "binary") {
			size_t nPosi = 0, nNega = 0;
			HistoGRAM *histo = y->hDistri != nullptr ? y->hDistri->histo : nullptr;		assert(histo!=nullptr);
			if (histo != nullptr) {
				assert(histo->nBins==2 || histo->nBins == 3);
				nPosi = histo->bins[1].nz;		nNega = histo->bins[0].nz;
				printf("\tNumber of positive : %lld, number of negative : %lld\n", nPosi, nNega);
			}
		}
		//printf("********* EDA::Analysis......OK\n");

	}
	//hFold->lossy.InitScore_(config);

}

void LambdaRank::Init(double sigma_, double a_0_, double a_1_, size_t nMost_, int flag) {
	printf("\n---- LambdaRank::Init..." );
	if (tables != nullptr) {
		if (a_0 == a_0_ && a_1 == a_1_ && sigma == sigma_ && nMost_ == nMost)
			return;

		delete[] tables;
	}
	sigma = sigma_;
	nMost = nMost_;
	a_0 = a_0_,					a_1 = a_1_;
	grid = (a_1 - a_0) / nMost;
	tables = new double[nMost];
	size_t i;
	double rou,a,fMin = DBL_MAX,fMax=-DBL_MAX;
	for (i = 0; i < nMost; i++) {
		a= a_0+grid*i;
		rou = 1 / (1 + exp(sigma*a));
		fMin = MIN2(rou, fMin);			fMax = MAX2(rou, fMax);
		tables[i] = rou;
	}
	printf("\n---- LambdaRank::Init sigma=%g a=[%.3g:-%.3g] rou=[%.3g:-%.3g]\n", sigma,a_0, a_1, fMin, fMax);
}

double LambdaRank::At(double a) {
	if (a < a_0) {
		nUnderFlow++;
		return tables[0];
	}	else if (a > a_1) {
		nOverFlow++;
		return tables[nMost-1];
	}
	else {
		int pos = (a - a_0) / grid;
		return tables[pos];

	}
}
/*
参见	gradient_boosting.py-(SKLearn)
LOSS_FUNCTIONS = {
	'ls': LeastSquaresError,
	'lad': LeastAbsoluteError,
	'huber': HuberLossFunction,
	'quantile': QuantileLossFunction,
	'deviance': None,    # for both, multinomial and binomial
	'exponential': ExponentialLoss,
}

class LeastSquaresError(RegressionLossFunction):
"""Loss function for least squares (LS) estimation.
Terminal regions need not to be updated for least squares. """
def init_estimator(self):
return MeanEstimator()

def __call__(self, y, pred, sample_weight=None):
if sample_weight is None:
return np.mean((y - pred.ravel()) ** 2.0)
else:
return (1.0 / sample_weight.sum() *
np.sum(sample_weight * ((y - pred.ravel()) ** 2.0)))

def negative_gradient(self, y, pred, **kargs):
return y - pred.ravel()

def update_terminal_regions(self, tree, X, y, residual, y_pred,
sample_weight, sample_mask,
lerning_rate=1.0, k=0):
"""Least squares does not need to update terminal regions.

But it has to update the predictions.
"""
# update predictions
y_pred[:, k] += lerning_rate * tree.predict(X).ravel()

def _update_terminal_region(self, tree, terminal_regions, leaf, X, y,
residual, pred, sample_weight):
pass


class BinomialDeviance(ClassificationLossFunction):
"""Binomial deviance loss function for binary classification.

Binary classification is a special case; here, we only need to
fit one tree instead of ``n_classes`` trees.
"""
def __init__(self, n_classes):
if n_classes != 2:
raise ValueError("{0:s} requires 2 classes.".format(
self.__class__.__name__))
# we only need to fit one tree for binary clf.
super(BinomialDeviance, self).__init__(1)

def init_estimator(self):
return LogOddsEstimator()

def __call__(self, y, pred, sample_weight=None):
"""Compute the deviance (= 2 * negative log-likelihood). """
# logaddexp(0, v) == log(1.0 + exp(v))
pred = pred.ravel()
if sample_weight is None:
return -2.0 * np.mean((y * pred) - np.logaddexp(0.0, pred))
else:
return (-2.0 / sample_weight.sum() *
np.sum(sample_weight * ((y * pred) - np.logaddexp(0.0, pred))))

def negative_gradient(self, y, pred, **kargs):
"""Compute the residual (= negative gradient). """
return y - expit(pred.ravel())

def _update_terminal_region(self, tree, terminal_regions, leaf, X, y,
residual, pred, sample_weight):
"""Make a single Newton-Raphson step.

our node estimate is given by:

sum(w * (y - prob)) / sum(w * prob * (1 - prob))

we take advantage that: y - prob = residual
"""
terminal_region = np.where(terminal_regions == leaf)[0]
residual = residual.take(terminal_region, axis=0)
y = y.take(terminal_region, axis=0)
sample_weight = sample_weight.take(terminal_region, axis=0)

numerator = np.sum(sample_weight * residual)
denominator = np.sum(sample_weight * (y - residual) * (1 - y + residual))

# prevents overflow and division by zero
if abs(denominator) < 1e-150:
tree.value[leaf, 0, 0] = 0.0
else:
tree.value[leaf, 0, 0] = numerator / denominator

def _score_to_proba(self, score):
proba = np.ones((score.shape[0], 2), dtype=np.float64)
proba[:, 1] = expit(score.ravel())
proba[:, 0] -= proba[:, 1]
return proba

def _score_to_decision(self, score):
proba = self._score_to_proba(score)
return np.argmax(proba, axis=1)
*/

/*
https://github.com/rushter/MLAlgorithms/blob/master/mla/ensemble/gbm.py
"""
References:
https://arxiv.org/pdf/1603.02754v3.pdf
http://www.saedsayad.com/docs/xgboost.pdf
https://homes.cs.washington.edu/~tqchen/pdf/BoostedTree.pdf
http://stats.stackexchange.com/questions/202858/loss-function-approximation-with-taylor-expansion
"""


class Loss:
"""Base class for loss functions."""

def __init__(self, regularization=1.0):
self.regularization = regularization

def grad(self, actual, predicted):
"""First order gradient."""
raise NotImplementedError()

def hess(self, actual, predicted):
"""Second order gradient."""
raise NotImplementedError()

def approximate(self, actual, predicted):
"""Approximate leaf value."""
return self.grad(actual, predicted).sum() / (self.hess(actual, predicted).sum() + self.regularization)

def transform(self, pred):
"""Transform predictions values."""
return pred

def gain(self, actual, predicted):
"""Calculate gain for split search."""
nominator = self.grad(actual, predicted).sum() ** 2
denominator = (self.hess(actual, predicted).sum() + self.regularization)
return 0.5 * (nominator / denominator)


class LeastSquaresLoss(Loss):
"""Least squares loss"""

def grad(self, actual, predicted):
return actual - predicted

def hess(self, actual, predicted):
return np.ones_like(actual)


class LogisticLoss(Loss):
"""Logistic loss."""

def grad(self, actual, predicted):
return actual * expit(-actual * predicted)

def hess(self, actual, predicted):
expits = expit(predicted)
return expits * (1 - expits)

def transform(self, output):
# Apply logistic (sigmoid) function to the output
return expit(output)
*/