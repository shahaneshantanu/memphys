# Author: Dr. Shantanu Shahane
OBJS = general_functions.o parameters.o points.o coefficient_computations.o cloud.o solver.o postprocessing_functions.o solidification.o
CC = mpicxx
DEBUG = -g
HYPRE_DIR = ../../../C_libraries/hypre-2.11.1/src/hypre#HYPRE
Eigen_directory = ../../../C_libraries/Eigen_3_3_7/ #Eigen library location
spectra_directory = ../../../C_libraries/spectra-0.8.1/include/ #Eigen library location
CFLAGS = -std=c++17 -Ofast -Wno-unused-result -march=native -I$(Eigen_directory) -I$(spectra_directory) -I$(HYPRE_DIR)/include -DHAVE_CONFIG_H -DHYPRE_TIMING -c
EFLAGS = -std=c++17
PROFLAGS = -pg #for code profiling
METISLIBS = ../../header_files/libmetis.a
LIBS      = -L$(HYPRE_DIR)/lib -lHYPRE -lm
LFLAGS    = $(LIBS) -lstdc++
INCLUDE = ../../header_files

########### Dependencies started
general_functions.o : ../../header_files/general_functions.cpp ../../header_files/general_functions.hpp
	$(CC) $(CFLAGS) ../../header_files/general_functions.cpp

parameters.o : ../../header_files/parameters.cpp ../../header_files/class.hpp ../../header_files/general_functions.hpp
	$(CC) $(CFLAGS) ../../header_files/parameters.cpp

points.o : ../../header_files/points.cpp ../../header_files/class.hpp ../../header_files/general_functions.hpp
	$(CC) $(CFLAGS) ../../header_files/points.cpp

solver.o : ../../header_files/solver.cpp ../../header_files/class.hpp ../../header_files/general_functions.hpp
	$(CC) $(CFLAGS) ../../header_files/solver.cpp

coefficient_computations.o : ../../header_files/coefficient_computations.cpp ../../header_files/class.hpp ../../header_files/general_functions.hpp
	$(CC) $(CFLAGS) ../../header_files/coefficient_computations.cpp

solidification.o : ../../header_files/solidification.cpp ../../header_files/class.hpp ../../header_files/general_functions.hpp
	$(CC) $(CFLAGS) ../../header_files/solidification.cpp

postprocessing_functions.o : ../../header_files/postprocessing_functions.cpp ../../header_files/class.hpp ../../header_files/general_functions.hpp
	$(CC) $(CFLAGS) ../../header_files/postprocessing_functions.cpp

cloud.o : ../../header_files/cloud.cpp ../../header_files/class.hpp ../../header_files/general_functions.hpp
	$(CC) $(CFLAGS) ../../header_files/cloud.cpp
########### Dependencies completed

########### Individual code files

solidification_problem_1.o : solidification_problem_1.cpp ../../header_files/class.hpp ../../header_files/general_functions.hpp
	$(CC) $(CFLAGS) $(LFLAGS) $(LIBS) -I $(INCLUDE) solidification_problem_1.cpp
solidification_problem_1 : $(OBJS) solidification_problem_1.o
	$(CC) $(PROFLAGS) -o out solidification_problem_1.o $(OBJS) $(LIBS)

########### Individual code completed

clean:
	rm -f *.o out *.out
rm_sim_files:
	find . -maxdepth 1 ! -name 'parameters_file.csv' -name '*.csv' -delete
	rm -f *.plt