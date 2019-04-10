#pragma once

#include <memory>
#include <string>
#include <vector>
#include "../util/GST_def.h"
#include "../include/LiteBOM_config.h"

using namespace std;

namespace Grusoft {
	template<typename Tx>
	void Imputer_Fill_(LiteBOM_Config&config, size_t nSamp_, Tx *vec,double fill,int flag=0x0 ) {
		for (size_t i = 0; i<nSamp_; i++) {
			if (IS_NAN_INF(vec[i])) {
				vec[i] = fill;
			}
		}
	}
}