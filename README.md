# AI for AE Path Planning and Graph Search
Path Planning and Graph Search Algorithms for AI for AE class by Tsiotras (Georgia Tech)

## Overview

Three modules are included, used for completing homework 2.

These include:
* 2D Astar solver for a grid based start to goal problem.
* Loyd's Eight-Puzzle disjoint set check and Astar graph solver using difference and manhatten distances heuristics
* Geometry Puzzle problem setup and visualization (See homework file)

### Astar

Modified structure from REDBLOBGAMES in Python.  Solves 2D path planning around walls

### Eight Puzzle

Created algorithm for determining the disjoint set (and thus available goal state) for any initial state.  
Modified graph search code to an Astar solver for the eight-puzzle problem using two heuristics:
* Total count of incorrect numbers in cells
* Sum of manhattan distance errors for each number

### Geometry Puzzle

Setup solving process for a geometry puzzle problem.  The goal is to fit all the given shapes into the required given goal shape.

