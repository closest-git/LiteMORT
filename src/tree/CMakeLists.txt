cmake_minimum_required(VERSION 3.5)

project(tree_)

# Create the main executable
# add the header files also so that the IDE knows they exist 
# in the source tree
add_library(tree_
	./ManifoldTree.cpp
	./GBRT.cpp
	./BoostingForest.cpp
	./BiSplit.cpp
	)