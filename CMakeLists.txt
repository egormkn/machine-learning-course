cmake_minimum_required(VERSION 3.13)
project(machine_learning_course)

set(CMAKE_CXX_STANDARD 17)

add_compile_options("$<$<CONFIG:DEBUG>:-DDEBUG>")

add_subdirectory(01-cross-validation)
add_subdirectory(02-f1-score)
add_subdirectory(03-metric-based-classification)
add_subdirectory(04-svm)
add_subdirectory(05-linear-regression)
add_subdirectory(06-decision-tree)
add_subdirectory(07-naive-bayes)
add_subdirectory(08-deep-neural-network)
add_subdirectory(09-logic-regression)
add_subdirectory(10-chi-square)
add_subdirectory(11-pearson-correlation)
add_subdirectory(12-distance)
add_subdirectory(13-conditional-entropy)
add_subdirectory(14-spearman-rank-correlation-coefficient)
add_subdirectory(15-conditional-variance)