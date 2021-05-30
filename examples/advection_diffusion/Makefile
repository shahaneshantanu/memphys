# Author: Dr. Shantanu Shahane
OBJS = general_functions.o parameters.o points.o coefficient_computations.o cloud.o solver.o postprocessing_functions.o
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

postprocessing_functions.o : ../../header_files/postprocessing_functions.cpp ../../header_files/class.hpp ../../header_files/general_functions.hpp
	$(CC) $(CFLAGS) ../../header_files/postprocessing_functions.cpp

cloud.o : ../../header_files/cloud.cpp ../../header_files/class.hpp ../../header_files/general_functions.hpp
	$(CC) $(CFLAGS) ../../header_files/cloud.cpp
########### Dependencies completed

########### Individual code files

heat_conduction.o : heat_conduction.cpp ../../header_files/class.hpp ../../header_files/general_functions.hpp
	$(CC) $(CFLAGS) $(LFLAGS) $(LIBS) -I $(INCLUDE) heat_conduction.cpp
heat_conduction : $(OBJS) heat_conduction.o
	$(CC) $(PROFLAGS) -o out heat_conduction.o $(OBJS) $(LIBS)

heat_conduction_periodic.o : heat_conduction_periodic.cpp ../../header_files/class.hpp ../../header_files/general_functions.hpp
	$(CC) $(CFLAGS) $(LFLAGS) $(LIBS) -I $(INCLUDE) heat_conduction_periodic.cpp
heat_conduction_periodic : $(OBJS) heat_conduction_periodic.o
	$(CC) $(PROFLAGS) -o out heat_conduction_periodic.o $(OBJS) $(LIBS)

heat_conduction_manuf_soln.o : heat_conduction_manuf_soln.cpp ../../header_files/class.hpp ../../header_files/general_functions.hpp
	$(CC) $(CFLAGS) $(LFLAGS) $(LIBS) -I $(INCLUDE) heat_conduction_manuf_soln.cpp
heat_conduction_manuf_soln : $(OBJS) heat_conduction_manuf_soln.o
	$(CC) $(PROFLAGS) -o out heat_conduction_manuf_soln.o $(OBJS) $(LIBS)

heat_conduction_manuf_soln_periodic.o : heat_conduction_manuf_soln_periodic.cpp ../../header_files/class.hpp ../../header_files/general_functions.hpp
	$(CC) $(CFLAGS) $(LFLAGS) $(LIBS) -I $(INCLUDE) heat_conduction_manuf_soln_periodic.cpp
heat_conduction_manuf_soln_periodic : $(OBJS) heat_conduction_manuf_soln_periodic.o
	$(CC) $(PROFLAGS) -o out heat_conduction_manuf_soln_periodic.o $(OBJS) $(LIBS)

heat_conduction_manuf_soln_SOR.o : heat_conduction_manuf_soln_SOR.cpp ../../header_files/class.hpp ../../header_files/general_functions.hpp
	$(CC) $(CFLAGS) $(LFLAGS) $(LIBS) -I $(INCLUDE) heat_conduction_manuf_soln_SOR.cpp
heat_conduction_manuf_soln_SOR : $(OBJS) heat_conduction_manuf_soln_SOR.o
	$(CC) $(PROFLAGS) -o out heat_conduction_manuf_soln_SOR.o $(OBJS) $(LIBS)

euler_steady_heat_cond_manuf_sol.o : euler_steady_heat_cond_manuf_sol.cpp ../../header_files/class.hpp ../../header_files/general_functions.hpp
	$(CC) $(CFLAGS) $(LFLAGS) $(LIBS) -I $(INCLUDE) euler_steady_heat_cond_manuf_sol.cpp
euler_steady_heat_cond_manuf_sol : $(OBJS) euler_steady_heat_cond_manuf_sol.o
	$(CC) $(PROFLAGS) -o out euler_steady_heat_cond_manuf_sol.o $(OBJS) $(LIBS)

euler_unsteady_heat_cond_manuf_sol.o : euler_unsteady_heat_cond_manuf_sol.cpp ../../header_files/class.hpp ../../header_files/general_functions.hpp
	$(CC) $(CFLAGS) $(LFLAGS) $(LIBS) -I $(INCLUDE) euler_unsteady_heat_cond_manuf_sol.cpp
euler_unsteady_heat_cond_manuf_sol : $(OBJS) euler_unsteady_heat_cond_manuf_sol.o
	$(CC) $(PROFLAGS) -o out euler_unsteady_heat_cond_manuf_sol.o $(OBJS) $(LIBS)

ab2_unsteady_heat_cond_manuf_sol.o : ab2_unsteady_heat_cond_manuf_sol.cpp ../../header_files/class.hpp ../../header_files/general_functions.hpp
	$(CC) $(CFLAGS) $(LFLAGS) $(LIBS) -I $(INCLUDE) ab2_unsteady_heat_cond_manuf_sol.cpp
ab2_unsteady_heat_cond_manuf_sol : $(OBJS) ab2_unsteady_heat_cond_manuf_sol.o
	$(CC) $(PROFLAGS) -o out ab2_unsteady_heat_cond_manuf_sol.o $(OBJS) $(LIBS)

scalar_transport_implicit.o : scalar_transport_implicit.cpp ../../header_files/class.hpp ../../header_files/general_functions.hpp
	$(CC) $(CFLAGS) $(LFLAGS) $(LIBS) -I $(INCLUDE) scalar_transport_implicit.cpp
scalar_transport_implicit : $(OBJS) scalar_transport_implicit.o
	$(CC) $(PROFLAGS) -o out scalar_transport_implicit.o $(OBJS) $(LIBS)

########### Individual code completed

clean:
	rm -f *.o out *.out
rm_sim_files:
	find . -maxdepth 1 ! -name 'parameters_file.csv' -name '*.csv' -delete
	rm -f *.plt