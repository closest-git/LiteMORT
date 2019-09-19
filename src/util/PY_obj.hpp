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

	bool isChar() {
		std::string type = dtype;
		return type == "char" || type == "int8";
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
			G_MEMCOPY_(nSamp, dst, (int8_t*)data, flag);
		}
	}
};


