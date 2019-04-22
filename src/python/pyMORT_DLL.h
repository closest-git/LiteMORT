// 下列 ifdef 块是创建使从 DLL 导出更简单的
// 宏的标准方法。此 DLL 中的所有文件都是用命令行上定义的 PYMORT_DLL_EXPORTS
// 符号编译的。在使用此 DLL 的
// 任何其他项目上不应定义此符号。这样，源文件中包含此文件的任何其他项目都会将
// PYMORT_DLL_API 函数视为是从 DLL 导入的，而此 DLL 则将用此宏定义的
// 符号视为是被导出的。
#ifdef PYMORT_DLL_EXPORTS
#define PYMORT_DLL_API __declspec(dllexport)
#else
#define PYMORT_DLL_API __declspec(dllimport)
#endif
#include "../util/GST_def.h"

#define __API_BEGIN__() try {

#define __API_END__() } \
catch(std::exception& ex) { return (ex); } \
catch(std::string& ex) { return (ex); } \
catch(...) { return ("unknown exception"); } \
return 0;


struct PY_ITEM {
	char *Keys;
	float Values;
	char *text;
	//int Index;
};


#ifdef __cplusplus
extern "C" {
#endif
	PYMORT_DLL_API void* LiteMORT_init(PY_ITEM* params, int nParam, int flag);
	PYMORT_DLL_API void LiteMORT_clear(void*);
	//PYMORT_DLL_API void LiteMORT_set_feat(PY_ITEM* params, int nParam, int flag);
	PYMORT_DLL_API void LiteMORT_fit(void *,float *h_data, tpY *h_target, size_t nSamp, size_t ldS, float *eval_data, tpY *eval_target, size_t nEval, size_t flag);
	PYMORT_DLL_API void LiteMORT_predict(void *,float *X, tpY *y, size_t nFeat_0, size_t nSamp, size_t flag);
	PYMORT_DLL_API void LiteMORT_Imputer_f(float *X, tpY *y, size_t nFeat_0, size_t nSamp, size_t flag);
	PYMORT_DLL_API void LiteMORT_Imputer_d(double *X, tpY *y, size_t nFeat_0, size_t nSamp, size_t flag);
	PYMORT_DLL_API void LiteMORT_EDA(void *, const float *X, const tpY *y, const size_t nFeat_0, const size_t nn, const size_t nValid,
		PY_ITEM* params, int nParam, const size_t flag);
#ifdef __cplusplus
}
#endif
