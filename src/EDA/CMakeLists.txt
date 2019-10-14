cmake_minimum_required(VERSION 3.5)

project(eda_)

# Create the main executable
# add the header files also so that the IDE knows they exist 
# in the source tree
add_library(eda_
	./Feat_Selection.cpp
	./SA_salp.cpp
	)