#pragma once
#include <chrono>
#include <thread>

#ifdef WIN32
	/*#include <time.h>
	#define GST_NOW( )		(clock( ))
	#define GST_TIC(tick)	clock_t tick=clock( );
	#define GST_TOC(tick)	((clock()-(tick))*1.0f/CLOCKS_PER_SEC)*/
	typedef std::chrono::high_resolution_clock Clock;
	#define GST_NOW( )	(Clock::now( ))
	#define GST_TIC(tick)	auto tick = Clock::now( );
	#define GST_TOC(tick)	( (std::chrono::duration_cast<std::chrono::microseconds>(Clock::now( )-(tick)).count( ))/1000000.0)

#else
	typedef std::chrono::high_resolution_clock Clock;
	#define GST_NOW( )	(Clock::now( ))
	#define GST_TIC(tick)	auto tick = Clock::now( );
	#define GST_TOC(tick)	( (std::chrono::duration_cast<std::chrono::microseconds>(Clock::now( )-(tick)).count( ))/1000000.0)
#endif