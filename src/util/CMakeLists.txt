cmake_minimum_required(VERSION 3.5)

project(util_)

# Create the main executable
# add the header files also so that the IDE knows they exist 
# in the source tree
add_library(util_
	./GRander.cpp
	)