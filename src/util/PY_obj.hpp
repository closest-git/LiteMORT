#pragma once

#include <memory>		//for shared_ptr
#include <string>
#include <vector>
#include <typeinfo>
#include <algorithm>
#include <complex>
#include <limits.h>
#include <cstring>
#include <stdio.h>  
#include "Object.hpp"

struct PY_COLUMN {
	char *name;
	void *data;
	char *dtype;
	char *type_x;
	double v_min;
	double v_max;

	bool isCategory() {
		return type_x!=NULL && strcmp(type_x,"*")==0;
	}
	bool isChar() {
		std::string type = dtype;
		return type == "char" || type == "int8" || type == "uint8";
	}
	bool isInt() {
		std::string type = dtype;
		return type == "int" || type == "int32";
	}
	bool isInt16() {
		std::string type = dtype;
		return type == "int16";
	}
	bool isInt64() {
		std::string type = dtype;
		return type == "int64";
	}
	bool isFloat() {
		std::string type = dtype;
		return type == "float32";
			/*|| type == "float16";	c++不支持，真可惜*/
	}
	bool isDouble() {
		std::string type = dtype;
		return type == "float64";
	}

	template<typename Tx>
	void CopyTo_(size_t nSamp, Tx* dst, int flag = 0x0) {
		if (isChar()) {
			//assert(typeof(Tx) == typeof(int8_t));
			//G_MEMCOPY_(nSamp, dst, (int8_t*)data, flag);
			int8_t *i8_ = (int8_t*)data;
			for (size_t i = 0; i < nSamp; i++) {
				dst[i] = i8_[i];
			}
		}
		else if (isDouble()){
			double *dbl = (double*)data;
			for (size_t i = 0; i < nSamp; i++) {
				dst[i] = dbl[i];
			}
		}	else if (isFloat()) {
			float *flt = (float*)data;
			for (size_t i = 0; i < nSamp; i++) {
				dst[i] = flt[i];
			}
		}	else if (isInt64()) {
			int64_t *i64 = (int64_t*)data;
			for (size_t i = 0; i < nSamp; i++) {
				dst[i] = i64[i];
			}
		}
		else if (isInt()) {
			int32_t *i32 = (int32_t*)data;
			for (size_t i = 0; i < nSamp; i++) {
				dst[i] = i32[i];
			}
		}
		else {
			throw "PY_COLUMN::CopyTo_ is mismatch!!!";
		}
	}
};


