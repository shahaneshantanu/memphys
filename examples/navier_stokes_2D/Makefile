# Author: Dr. Shantanu Shahane
OBJS = general_functions.o parameters.o points.o coefficient_computations.o cloud.o solver.o navier_stokes.o postprocessing_functions.o
CC = mpicxx
DEBUG = -g
HYPRE_DIR = ../../../C_libraries/hypre-2.11.1/src/hypre#HYPRE
Eigen_directory = ../../../C_libraries/Eigen_3_3_7/ #Eigen library location
spectra_directory = ../../../C_libraries/spectra-0.8.1/include/ #Eigen library location
CFLAGS = -std=c++17 -Ofast -Wno-unused-result -march=native -I$(Eigen_directory) -I$(spectra_directory) -I$(HYPRE_DIR)/include -DHAVE_CONFIG_H -DHYPRE_TIMING -c
EFLAGS = -std=c++17
PROFLAGS = -pg #for code profiling
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

navier_stokes.o : ../../header_files/navier_stokes.cpp ../../header_files/class.hpp ../../header_files/general_functions.hpp
	$(CC) $(CFLAGS) ../../header_files/navier_stokes.cpp

postprocessing_functions.o : ../../header_files/postprocessing_functions.cpp ../../header_files/class.hpp ../../header_files/general_functions.hpp
	$(CC) $(CFLAGS) ../../header_files/postprocessing_functions.cpp

cloud.o : ../../header_files/cloud.cpp ../../header_files/class.hpp ../../header_files/general_functions.hpp
	$(CC) $(CFLAGS) ../../header_files/cloud.cpp
########### Dependencies completed

########### Individual code files

kovasznay_flow.o : kovasznay_flow.cpp ../../header_files/class.hpp ../../header_files/general_functions.hpp
	$(CC) $(CFLAGS) $(LFLAGS) $(LIBS) -I $(INCLUDE) kovasznay_flow.cpp
kovasznay_flow : $(OBJS) kovasznay_flow.o
	$(CC) $(PROFLAGS) -o out kovasznay_flow.o $(OBJS) $(LIBS)

couette_flow.o : couette_flow.cpp ../../header_files/class.hpp ../../header_files/general_functions.hpp
	$(CC) $(CFLAGS) $(LFLAGS) $(LIBS) -I $(INCLUDE) couette_flow.cpp
couette_flow : $(OBJS) couette_flow.o
	$(CC) $(PROFLAGS) -o out couette_flow.o $(OBJS) $(LIBS)

eccentric_couette_flow.o : eccentric_couette_flow.cpp ../../header_files/class.hpp ../../header_files/general_functions.hpp
	$(CC) $(CFLAGS) $(LFLAGS) $(LIBS) -I $(INCLUDE) eccentric_couette_flow.cpp
eccentric_couette_flow : $(OBJS) eccentric_couette_flow.o
	$(CC) $(PROFLAGS) -o out eccentric_couette_flow.o $(OBJS) $(LIBS)

elliptic_couette_flow.o : elliptic_couette_flow.cpp ../../header_files/class.hpp ../../header_files/general_functions.hpp
	$(CC) $(CFLAGS) $(LFLAGS) $(LIBS) -I $(INCLUDE) elliptic_couette_flow.cpp
elliptic_couette_flow : $(OBJS) elliptic_couette_flow.o
	$(CC) $(PROFLAGS) -o out elliptic_couette_flow.o $(OBJS) $(LIBS)

flow_over_cylinder.o : flow_over_cylinder.cpp ../../header_files/class.hpp ../../header_files/general_functions.hpp
	$(CC) $(CFLAGS) $(LFLAGS) $(LIBS) -I $(INCLUDE) flow_over_cylinder.cpp
flow_over_cylinder : $(OBJS) flow_over_cylinder.o
	$(CC) $(PROFLAGS) -o out flow_over_cylinder.o $(OBJS) $(LIBS)

channel_flow.o : channel_flow.cpp ../../header_files/class.hpp ../../header_files/general_functions.hpp
	$(CC) $(CFLAGS) $(LFLAGS) $(LIBS) -I $(INCLUDE) channel_flow.cpp
channel_flow : $(OBJS) channel_flow.o
	$(CC) $(PROFLAGS) -o out channel_flow.o $(OBJS) $(LIBS)

taylor_green_modified.o : taylor_green_modified.cpp ../../header_files/class.hpp ../../header_files/general_functions.hpp
	$(CC) $(CFLAGS) $(LFLAGS) $(LIBS) -I $(INCLUDE) taylor_green_modified.cpp
taylor_green_modified : $(OBJS) taylor_green_modified.o
	$(CC) $(PROFLAGS) -o out taylor_green_modified.o $(OBJS) $(LIBS)

taylor_green_steady.o : taylor_green_steady.cpp ../../header_files/class.hpp ../../header_files/general_functions.hpp
	$(CC) $(CFLAGS) $(LFLAGS) $(LIBS) -I $(INCLUDE) taylor_green_steady.cpp
taylor_green_steady : $(OBJS) taylor_green_steady.o
	$(CC) $(PROFLAGS) -o out taylor_green_steady.o $(OBJS) $(LIBS)

########### Individual code completed

clean:
	rm -f *.o out *.out
