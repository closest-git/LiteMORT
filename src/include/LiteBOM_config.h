#pragma once

#include <string>
#include <iostream>

namespace Grusoft {

	using std::cout;
	using std::endl;
	using std::string;

	class LiteBOM_Config {
	public:
		enum TaskType {
			kTrain, kPredict, kConvertModel, KRefitTree
		};
		LiteBOM_Config() {}

		LiteBOM_Config(int _num_trees,
			int _max_leaf,
			double _min_gain,
			double _l2_reg,
			std::string _loss,
			double _learning_rate,
			std::string _eval_metric,
			int _num_classes,
			int _num_threads,
			//int _max_bin,
			double _min_sum_hessian_in_leaf,
			int _num_vars,
			std::string _grow_by,
			std::string _leaf_type,
			int _max_level,
			int _verbose,
			double _sparse_threshold,
			string _boosting_type,
			double _goss_alpha,
			double _goss_beta) :
			num_trees(_num_trees),
			//max_leaf(_max_leaf),
			min_gain(_min_gain),
			loss(_loss),
			learning_rate(_learning_rate),
			eval_metric(_eval_metric),
			num_classes(_num_classes),
			num_threads(_num_threads),
			//max_bin(_max_bin),
			min_sum_hessian_in_leaf(_min_sum_hessian_in_leaf),
			num_vars(_num_vars),
			grow_by(_grow_by),
			max_level(_max_level),
			l2_reg(_l2_reg),
			verbose(_verbose),
			leaf_type(_leaf_type),
			sparse_threshold(_sparse_threshold),
			boosting_type(_boosting_type),
			goss_alpha(_goss_alpha),
			goss_beta(_goss_beta)
		{

			all_prepare_bin_time = 0.0;
			all_prepare_histogram_time = 0.0;
			all_find_split_time = 0.0;
			all_split_time = 0.0;
			all_update_train_score_time = 0.0;
			all_update_gradients_time = 0.0;
			all_after_train_tree_time = 0.0;

			prepare_bin_time = 0.0;
			prepare_histogram_time = 0.0;
			find_split_time = 0.0;
			split_time = 0.0;
			update_train_score_time = 0.0;
			update_gradients_time = 0.0;
			after_train_tree_time = 0.0;

			cout << "num_trees " << num_trees << endl;
			//cout << "max_leaf " << max_leaf << endl;
			cout << "min_gain " << min_gain << endl;
			cout << "l2_reg " << l2_reg << endl;
			cout << "loss " << loss << endl;
			cout << "learning_rate " << learning_rate << endl;
			cout << "eval_metric " << eval_metric << endl;
			cout << "num_classes " << num_classes << endl;
			cout << "num_threads " << num_threads << endl;
			//cout << "max_bin " << _max_bin << endl;
			cout << "min_sum_hessian_in_leaf " << min_sum_hessian_in_leaf << endl;
			cout << "num_vars " << num_vars << endl;
			cout << "grow_by " << grow_by << endl;
			cout << "max_level " << max_level << endl;
			cout << "verbose " << verbose << endl;
			cout << "leaf type " << _leaf_type << endl;
			cout << "sparse threshold 1 " << sparse_threshold << endl;
			cout << "boosting_type " << boosting_type << endl;
			cout << "goss_alpha " << goss_alpha << endl;
			cout << "goss_beta " << goss_beta << endl;
		}

		int num_trees;

		//int max_leaf;

		double min_gain;

		double l2_reg;

		std::string loss;

		//double learning_rate;

		std::string eval_metric;

		int num_classes;

		bool train_multi_class;

		//int num_threads;

		//int max_bin;
		int min_data_in_bin = 8;

		//double min_sum_hessian_in_leaf;

		int num_vars;

		int max_level;

		int verbose=0;
		int verbose_eval = 200;

		std::string grow_by;

		std::string leaf_type;

		double sparse_threshold;

		double all_prepare_bin_time;
		double prepare_bin_time;
		double all_prepare_histogram_time;
		double prepare_histogram_time;
		double all_find_split_time;
		double find_split_time;
		double all_update_gradients_time;
		double prepare_local_data_time;
		double all_split_time;
		double split_time;
		double update_train_score_time;
		double all_update_train_score_time;
		double update_gradients_time;
		double all_after_train_tree_time;
		double after_train_tree_time;

		string boosting_type;
		double goss_alpha;
		double goss_beta;

		void dump(int type = 0x0) {
			if (verbose <= 0)
				return;

			//string histo_alg = histo_algorithm==on_EDA ? "on_EDA" : histo_algorithm == on_subsample ? "on_subsample" : "on_Y";
			string histo_alg = histo_bin_map == 0 ? "\"quantile\"" : (histo_bin_map == 1 ? "\"frequency\"" : histo_bin_map == 3 ? "\"Dcrimini on Y\"" : "XXX");
			printf("\n\t%slr=%g sample=[%g,%g] min@leaf=%d stop=%d drop=%g num_leaves=%d feat_quanti=%d"
				"\n\tOBJECTIVE=\"%s\"\teval_metric=\"%s\"\tleaf_optimal=\"%s\""
				"\n\t init=%s maxDepth=%d"
				"\n\tL2=%.8g\tLf=%.8g\tImputation=%s\tNormal=%s"
				"\n\tnElitism=%g,Iter_refine=%g \tRefine_split=%d"
				"\n\tnMostPrune=%d node_task=%s debug=%s"
				"\n\tnMostSalp4Bins=%d histo_bin_::map=%s",
				lr_adptive_leaf?"a_":"",learning_rate, subsample, feature_fraction, min_data_in_leaf, early_stopping_round, drop_out, num_leaves, feat_quanti,
				objective.c_str(), eval_metric.c_str(), leaf_optimal.c_str(),
				init_scor.c_str(), max_depth,
				lambda_l2, lambda_Feat,eda_NA == -1 ? "OFF" : eda_NA == 0 ? "0" : "Other",
				eda_Normal == 0 ? "OFF" : "Gaussian", 
				rElitism,T_iterrefine,split_refine,
				nMostPrune,node_task==0 ? "split_X" : node_task == 1 ? "split_Y" : "REGRESS_X", isDebug_1?"Debug_1":"Debug_0",
				nMostSalp4bins,histo_alg.c_str()
				);
		}
		
		#pragma region Core Parameters

		// [doc-only]
		// alias = config_file
		// desc = path of config file
		// desc = **Note**: can be used only in CLI version
		std::string config = "";

		// [doc-only]
		// type = enum
		// default = train
		// options = train, predict, convert_model, refit
		// alias = task_type
		// desc = ``train``, for training, aliases: ``training``
		// desc = ``predict``, for prediction, aliases: ``prediction``, ``test``
		// desc = ``convert_model``, for converting model file into if-else format, see more information in `IO Parameters <#io-parameters>`__
		// desc = ``refit``, for refitting existing models with new data, aliases: ``refit_tree``
		// desc = **Note**: can be used only in CLI version
		TaskType task = TaskType::kTrain;

		// [doc-only]
		// type = enum
		// options = regression, regression_l1, huber, fair, poisson, quantile, mape, gammma, tweedie, binary, multiclass, multiclassova, xentropy, xentlambda, lambdarank
		// alias = objective_type, app, application
		// desc = regression application
		// descl2 = ``regression_l2``, L2 loss, aliases: ``regression``, ``mean_squared_error``, ``mse``, ``l2_root``, ``root_mean_squared_error``, ``rmse``
		// descl2 = ``regression_l1``, L1 loss, aliases: ``mean_absolute_error``, ``mae``
		// descl2 = ``huber``, `Huber loss <https://en.wikipedia.org/wiki/Huber_loss>`__
		// descl2 = ``fair``, `Fair loss <https://www.kaggle.com/c/allstate-claims-severity/discussion/24520>`__
		// descl2 = ``poisson``, `Poisson regression <https://en.wikipedia.org/wiki/Poisson_regression>`__
		// descl2 = ``quantile``, `Quantile regression <https://en.wikipedia.org/wiki/Quantile_regression>`__
		// descl2 = ``mape``, `MAPE loss <https://en.wikipedia.org/wiki/Mean_absolute_percentage_error>`__, aliases: ``mean_absolute_percentage_error``
		// descl2 = ``gamma``, Gamma regression with log-link. It might be useful, e.g., for modeling insurance claims severity, or for any target that might be `gamma-distributed <https://en.wikipedia.org/wiki/Gamma_distribution#Applications>`__
		// descl2 = ``tweedie``, Tweedie regression with log-link. It might be useful, e.g., for modeling total loss in insurance, or for any target that might be `tweedie-distributed <https://en.wikipedia.org/wiki/Tweedie_distribution#Applications>`__
		// desc = ``binary``, binary `log loss <https://en.wikipedia.org/wiki/Cross_entropy>`__ classification (or logistic regression). Requires labels in {0, 1}; see ``xentropy`` for general probability labels in [0, 1]
		// desc = multi-class classification application
		// descl2 = ``multiclass``, `softmax <https://en.wikipedia.org/wiki/Softmax_function>`__ objective function, aliases: ``softmax``
		// descl2 = ``multiclassova``, `One-vs-All <https://en.wikipedia.org/wiki/Multiclass_classification#One-vs.-rest>`__ binary objective function, aliases: ``multiclass_ova``, ``ova``, ``ovr``
		// descl2 = ``num_class`` should be set as well
		// desc = cross-entropy application
		// descl2 = ``xentropy``, objective function for cross-entropy (with optional linear weights), aliases: ``cross_entropy``
		// descl2 = ``xentlambda``, alternative parameterization of cross-entropy, aliases: ``cross_entropy_lambda``
		// descl2 = label is anything in interval [0, 1]
		// desc = ``lambdarank``, `lambdarank <https://papers.nips.cc/paper/2971-learning-to-rank-with-nonsmooth-cost-functions.pdf>`__ application
		// descl2 = label should be ``int`` type in lambdarank tasks, and larger number represents the higher relevance (e.g. 0:bad, 1:fair, 2:good, 3:perfect)
		// descl2 = `label_gain <#objective-parameters>`__ can be used to set the gain (weight) of ``int`` label
		// descl2 = all values in ``label`` must be smaller than number of elements in ``label_gain``
		std::string objective = "regression";

		virtual void OnObjective();

		// [doc-only]
		// type = enum
		// alias = boosting_type, boost
		// options = gbdt, gbrt, rf, random_forest, dart, goss
		// desc = ``gbdt``, traditional Gradient Boosting Decision Tree, aliases: ``gbrt``
		// desc = ``rf``, Random Forest, aliases: ``random_forest``
		// desc = ``dart``, `Dropouts meet Multiple Additive Regression Trees <https://arxiv.org/abs/1505.01866>`__
		// desc = ``goss``, Gradient-based One-Side Sampling
		std::string boosting = "gbdt";

		// alias = train, train_data, data_filename
		std::string data = "";

		// alias = test, valid_data, valid_data_file, test_data, valid_filenames
		std::vector<std::string> valid;

		// alias = num_iteration, num_tree, num_trees, num_round, num_rounds, num_boost_round, n_estimators
		int num_iterations = 10;

		// alias = shrinkage_rate
		// check = >0.0
		// desc = shrinkage rate
		// desc = in ``dart``, it also affects on normalization weights of dropped trees
		double learning_rate = 0.03;
		bool isDebug_1 = true;
		bool isDynamicHisto = false;
		bool lr_adptive_leaf = false;
		int adaptive_sample_weight = 0;
		bool useRandomSeed = false;
		std::string eda_nan = "zero";
		int nMostSalp4bins = 0;
		int nMostSalp4feats = 0;
		int nMostPrune = 0;
		double rElitism = 0.05;
		float *feat_selector = nullptr;

		//gradient variance
		//std::string leaf_optimal = "taylor_2";		//似乎有问题，grad_variance收敛速度明显更快
		std::string leaf_optimal = "grad_variance";	
		//std::string leaf_optimal = "ap_outlier";
		//std::string leaf_optimal = "lambda_0";

		std::string init_scor = "mean";		//or path of .init file
		//std::string init_scor = "0";		

		std::string leaf_regression = "none";
		//std::string leaf_regression = "linear";
		//std::string leaf_regression = "histo_mean";	//似乎过拟合，参见10_8_[histo_mean]_1.5771.info

		typedef enum{ on_QUANTILE,on_FREQ,   onUNIQUE, on_FREQ_and_Y,
		} HISTO_BINS_MAP;
		HISTO_BINS_MAP histo_bin_map= HISTO_BINS_MAP::on_FREQ;		//似乎on_FREQ更合理一些
		int _histo_bins_Y = 0;	//如果大于1，则自动update
		
		/*
		typedef enum { on_EDA, on_subsample, on_Y } HISTO_ALGORITHM;
		HISTO_ALGORITHM histo_algorithm;*/

		typedef enum { NA_=-1,NA_ZERO=0, NA_MEAN, NA_MEDIAN,NA_MOST		} EDA_NA;
		EDA_NA eda_NA;

		typedef enum { NORMAL_off = 0, 
			NORMAL_gaussian,		//Gaussian normalization typically leads to normalized x-values that are all generally between -10 and +10.
			NORMAL_min_max = 0, 
			NORMAL_z_score  
		} EDA_NORMAL;
		EDA_NORMAL eda_Normal= EDA_NORMAL::NORMAL_off;

		// default = 256,	some people said "Fewer than 256 bins in a histogram is enough to achieve good accuracy"
		int feat_quanti = 0;
		double T_iterrefine = 0;

		typedef enum {
			split_X = 0,
			histo_X_split_G,				//histo与X一样，但split依照Grad	
			REGRESS_X,
			REGRESS_Y,
			histo_Y_split_Y=111,
		} NODAL_TASK;
		NODAL_TASK node_task = split_X;		//0:split_X		1:split_Y		2:regression
											//int _histo_bins_Y=0;	//如果大于1，则自动update
		bool histo_bins_onY()	const; /*{
			//return _histo_bins_Y!=0;
			return node_task == split_Y;
		}*/

		typedef enum {
			REFINE_NONE = 0,
			TWO_PASS,				
			NEAR_CUT,
		} REFINE_SPLIT;
		REFINE_SPLIT split_refine = REFINE_NONE;
		// INT_MAX
		//int histo_bins = 0;
		// default = 31
		// alias = num_leaf
		// check = >1
		// desc = max number of leaves in one tree
		int num_leaves = 128;	

		// [doc-only]
		// type = enum
		// options = serial, feature, data, voting
		// alias = tree, tree_learner_type
		// desc = ``serial``, single machine tree learner
		// desc = ``feature``, feature parallel tree learner, aliases: ``feature_parallel``
		// desc = ``data``, data parallel tree learner, aliases: ``data_parallel``
		// desc = ``voting``, voting parallel tree learner, aliases: ``voting_parallel``
		// desc = refer to `Parallel Learning Guide <./Parallel-Learning-Guide.rst>`__ to get more details
		std::string tree_learner = "serial";

		int num_threads = -1;

		// [doc-only]
		// alias = random_seed
		// desc = this seed is used to generate other seeds, e.g. ``data_random_seed``, ``feature_fraction_seed``
		// desc = will be overridden, if you set other seeds
		int seed = 0;

#pragma region Learning Control Parameters

		// desc = limit the max depth for tree model. This is used to deal with over-fitting when ``#data`` is small. Tree still grows leaf-wise
		// desc = ``< 0`` means no limit
		int max_depth = -3;

		// alias = min_data_per_leaf, min_data, min_child_samples
		// check = >=0
		// desc = minimal number of data in one leaf. Can be used to deal with over-fitting
		int min_data_in_leaf = 50;

		// alias = min_sum_hessian_per_leaf, min_sum_hessian, min_hessian, min_child_weight
		// check = >=0.0
		// desc = minimal sum hessian in one leaf. Like ``min_data_in_leaf``, it can be used to deal with over-fitting
		double min_sum_hessian_in_leaf = 1e-3;

		// alias = sub_row, subsample, bagging,bagging_fraction
		// check = >0.0
		// check = <=1.0
		// desc = like ``feature_fraction``, but this will randomly select part of data without resampling
		// desc = can be used to speed up training
		// desc = can be used to deal with over-fitting
		// desc = **Note**: to enable bagging, ``bagging_freq`` should be set to a non zero value as well
		double subsample = 1;

		//1-25测试，没有效果（似乎batch的样本变小，其二分类的泛化效果就不好）
		double batch = 1;

		// alias = subsample_freq
		// desc = frequency for bagging
		// desc = ``0`` means disable bagging; ``k`` means perform bagging at every ``k`` iteration
		// desc = **Note**: to enable bagging, ``bagging_fraction`` should be set to value smaller than ``1.0`` as well
		int bagging_freq = 0;

		// alias = bagging_fraction_seed
		// desc = random seed for bagging
		int bagging_seed = 3;

		// alias = sub_feature, colsample_bytree
		double feature_fraction = 1.0;

		// desc = random seed for ``feature_fraction``
		int feature_fraction_seed = 2;

		// alias = early_stopping_rounds, early_stopping
		// desc = will stop training if one metric of one validation data doesn't improve in last ``early_stopping_round`` rounds
		// desc = ``<= 0`` means disable
		int early_stopping_round = 0;

		// alias = max_tree_output, max_leaf_output
		// desc = used to limit the max output of tree leaves
		// desc = ``<= 0`` means no constraint
		// desc = the final max output of leaves is ``learning_rate * max_delta_step``
		double max_delta_step = 0.0;

		// alias = reg_alpha
		// check = >=0.0
		// desc = L1 regularization
		double lambda_l1 = 0.0;
		double lambda_Feat = 1;	// 0.5;

		// alias = reg_lambda
		// check = >=0.0
		// desc = L2 regularization
		double lambda_l2 = 0.0;

		// alias = min_split_gain
		// check = >=0.0
		// desc = the minimal gain to perform split
		double min_gain_to_split = 0.0;

		// check = >=0.0
		// check = <=1.0
		// desc = used only in ``dart``
		// desc = dropout rate
		double drop_rate = 0.1;

		// desc = used only in ``dart``
		// desc = max number of dropped trees on one iteration
		// desc = ``<=0`` means no limit
		int max_drop = 50;

		// check = >=0.0
		// check = <=1.0
		// desc = used only in ``dart``
		// desc = probability of skipping drop
		double drop_out = 1;		//

		// desc = used only in ``dart``
		// desc = set this to ``true``, if you want to use xgboost dart mode
		bool xgboost_dart_mode = false;

		// desc = used only in ``dart``
		// desc = set this to ``true``, if you want to use uniform drop
		bool uniform_drop = false;

		// desc = used only in ``dart``
		// desc = random seed to choose dropping models
		int drop_seed = 4;

		// check = >=0.0
		// check = <=1.0
		// desc = used only in ``goss``
		// desc = the retain ratio of large gradient data
		double top_rate = 0.2;

		// check = >=0.0
		// check = <=1.0
		// desc = used only in ``goss``
		// desc = the retain ratio of small gradient data
		double other_rate = 0.1;

		// check = >0
		// desc = minimal number of data per categorical group
		int min_data_per_group = 100;

		// check = >0
		// desc = used for the categorical features
		// desc = limit the max threshold points in categorical features
		int max_cat_threshold = 32;

		// check = >=0.0
		// desc = used for the categorical features
		// desc = L2 regularization in categorcial split
		double cat_l2 = 10.0;

		// check = >=0.0
		// desc = used for the categorical features
		// desc = this can reduce the effect of noises in categorical features, especially for categories with few data
		double cat_smooth = 10.0;

		// check = >0
		// desc = when number of categories of one feature smaller than or equal to ``max_cat_to_onehot``, one-vs-other split algorithm will be used
		int max_cat_to_onehot = 4;

		// alias = topk
		// check = >0
		// desc = used in `Voting parallel <./Parallel-Learning-Guide.rst#choose-appropriate-parallel-algorithm>`__
		// desc = set this to larger value for more accurate result, but it will slow down the training speed
		int top_k = 20;

		// type = multi-int
		// alias = mc, monotone_constraint
		// default = None
		// desc = used for constraints of monotonic features
		// desc = ``1`` means increasing, ``-1`` means decreasing, ``0`` means non-constraint
		// desc = you need to specify all features in order. For example, ``mc=-1,0,1`` means decreasing for 1st feature, non-constraint for 2nd feature and increasing for the 3rd feature
		std::vector<int8_t> monotone_constraints;

		// type = multi-double
		// alias = fc, fp, feature_penalty
		// default = None
		// desc = used to control feature's split gain, will use ``gain[i] = max(0, feature_contri[i]) * gain[i]`` to replace the split gain of i-th feature
		// desc = you need to specify all features in order
		std::vector<double> feature_contri;

#pragma endregion

	#pragma endregion
	};

}   


