cmake_minimum_required(VERSION 3.5)

project(data_fold_)

# Create the main executable
# add the header files also so that the IDE knows they exist 
# in the source tree
add_library(data_fold_
	./Histogram.cpp
	./EDA.cpp
	./DataFold.cpp
	./FeatVector.cpp
	)