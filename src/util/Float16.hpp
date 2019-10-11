#pragma once
#include <memory.h>

//extern short FloatToFloat16(float value);
//extern float Float16ToFloat(short value);

class Float16
{
protected:
	short mValue;

	short FloatToFloat16(float value) {
		short   fltInt16;
		int     fltInt32;
		memcpy(&fltInt32, &value, sizeof(float));
		fltInt16 = ((fltInt32 & 0x7fffffff) >> 13) - (0x38000000 >> 13);
		fltInt16 |= ((fltInt32 & 0x80000000) >> 16);

		return fltInt16;
	}

	float Float16ToFloat(short fltInt16) const {
		int fltInt32 = ((fltInt16 & 0x8000) << 16);
		fltInt32 |= ((fltInt16 & 0x7fff) << 13) + 0x38000000;

		float fRet;
		memcpy(&fRet, &fltInt32, sizeof(float));
		return fRet;
	}

public:
	Float16();
	Float16(float value);
	Float16(const Float16& value);

	operator float();
	operator float() const;

	friend Float16 operator + (const Float16& val1, const Float16& val2);
	friend Float16 operator - (const Float16& val1, const Float16& val2);
	friend Float16 operator * (const Float16& val1, const Float16& val2);
	friend Float16 operator / (const Float16& val1, const Float16& val2);

	Float16& operator =(const Float16& val);
	Float16& operator +=(const Float16& val);
	Float16& operator -=(const Float16& val);
	Float16& operator *=(const Float16& val);
	Float16& operator /=(const Float16& val);
	Float16& operator -();

	//https://codedocs.xyz/HipsterSloth/PSMoveService/__detail_8hpp_source.html
	union uif32 {
		float f;
		unsigned int i;
	};

	//https://github.com/g-truc/glm/blob/0.9.5/glm/detail/type_half.inl
	static float GLM_toFloat32(const short& value, int flag = 0x0) {
		int s = (value >> 15) & 0x00000001;
		int e = (value >> 10) & 0x0000001f;
		int m = value & 0x000003ff;
		uif32 result;
		if (e == 0) {
			if (m == 0) {
				result.i = (unsigned int)(s << 31);
				return result.f;
			}
			else {
				//
				// Denormalized number -- renormalize it
				//
				while (!(m & 0x00000400)) {
					m <<= 1;
					e -= 1;
				}
				e += 1;
				m &= ~0x00000400;
			}
		}
		else if (e == 31) {
			if (m == 0) {
				//
				// Positive or negative infinity
				//
				result.i = (unsigned int)((s << 31) | 0x7f800000);
				return result.f;
			}		else {
				//
				// Nan -- preserve sign and significand bits
				//
				uif32 result;
				result.i = (unsigned int)((s << 31) | 0x7f800000 | (m << 13));
				return result.f;
			}
		}

		e = e + (127 - 15);
		m = m << 13;
		uif32 Result;
		Result.i = (unsigned int)((s << 31) | (e << 23) | m);
		return Result.f;
	};
};

/*+----+----+----+----+----+----+----+----+----+----+----+----+----+----+----+----+----+----+*/

inline Float16::Float16()
{
}

/*+----+----+----+----+----+----+----+----+----+----+----+----+----+----+----+----+----+----+*/

inline Float16::Float16(float value){
	mValue = FloatToFloat16(value);
}

/*+----+----+----+----+----+----+----+----+----+----+----+----+----+----+----+----+----+----+*/

inline Float16::Float16(const Float16 &value){
	mValue = value.mValue;
}

/*+----+----+----+----+----+----+----+----+----+----+----+----+----+----+----+----+----+----+*/

inline Float16::operator float()
{
	return Float16ToFloat(mValue);
}

/*+----+----+----+----+----+----+----+----+----+----+----+----+----+----+----+----+----+----+*/

inline Float16::operator float() const
{
	return Float16ToFloat(mValue);
}

/*+----+----+----+----+----+----+----+----+----+----+----+----+----+----+----+----+----+----+*/

inline Float16& Float16::operator =(const Float16& val)
{
	mValue = val.mValue;
}

/*+----+----+----+----+----+----+----+----+----+----+----+----+----+----+----+----+----+----+*/

inline Float16& Float16::operator +=(const Float16& val)
{
	*this = *this + val;
	return *this;
}

/*+----+----+----+----+----+----+----+----+----+----+----+----+----+----+----+----+----+----+*/

inline Float16& Float16::operator -=(const Float16& val)
{
	*this = *this - val;
	return *this;

}

/*+----+----+----+----+----+----+----+----+----+----+----+----+----+----+----+----+----+----+*/

inline Float16& Float16::operator *=(const Float16& val)
{
	*this = *this * val;
	return *this;
}

/*+----+----+----+----+----+----+----+----+----+----+----+----+----+----+----+----+----+----+*/

inline Float16& Float16::operator /=(const Float16& val)
{
	*this = *this / val;
	return *this;
}

/*+----+----+----+----+----+----+----+----+----+----+----+----+----+----+----+----+----+----+*/

inline Float16& Float16::operator -()
{
	*this = Float16(-(float)*this);
	return *this;
}

/*+----+----+----+----+----+----+----+----+----+----+----+----+----+----+----+----+----+----+*/
/*+----+                                 Friends                                       +----+*/
/*+----+----+----+----+----+----+----+----+----+----+----+----+----+----+----+----+----+----+*/

inline Float16 operator + (const Float16& val1, const Float16& val2)
{
	return Float16((float)val1 + (float)val2);
}

/*+----+----+----+----+----+----+----+----+----+----+----+----+----+----+----+----+----+----+*/

inline Float16 operator - (const Float16& val1, const Float16& val2)
{
	return Float16((float)val1 - (float)val2);
}

/*+----+----+----+----+----+----+----+----+----+----+----+----+----+----+----+----+----+----+*/

inline Float16 operator * (const Float16& val1, const Float16& val2)
{
	return Float16((float)val1 * (float)val2);
}

/*+----+----+----+----+----+----+----+----+----+----+----+----+----+----+----+----+----+----+*/

inline Float16 operator / (const Float16& val1, const Float16& val2)
{
	return Float16((float)val1 / (float)val2);
}




