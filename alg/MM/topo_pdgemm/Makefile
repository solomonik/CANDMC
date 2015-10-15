include ../../../config.mk

LIB_DIR = ../../../lib

NuMM: $(LIB_DIR)/libNuMM.a

$(LIB_DIR)/libNuMM.a: d25_summa.o summa.o dual_cannon.o

#topo_pdgemm_bench: topo_pdgemm_bench.cxx topo_pdgemm_algs.h summa.o util.o comm.o d25_summa.o dual_cannon.o omp_dgemm_ring.o
#	$(CXX) util.o comm.o summa.o d25_summa.o dual_cannon.o omp_dgemm_ring.o \
#	-o topo_pdgemm_bench topo_pdgemm_bench.cxx \
#	$(CXXFLAGS) $(DEFS)  \
#	$(LIBS) $(BLAS) $(LAPACK) $(BLAS) -lm
#
#topo_pdgemm_unit: topo_pdgemm_unit.cxx topo_pdgemm_algs.h summa.o util.o comm.o d25_summa.o dual_cannon.o omp_dgemm_ring.o
#	$(CXX) util.o comm.o summa.o d25_summa.o dual_cannon.o omp_dgemm_ring.o \
#	-o topo_pdgemm_unit topo_pdgemm_unit.cxx \
#	$(UFLAGS) $(DEFS)  \
#	$(LIBS) $(BLAS) $(LAPACK) $(BLAS) -lm

d25_summa.o: d25_summa.cxx topo_pdgemm_algs.h
	$(CXX) -c d25_summa.cxx -o d25_summa.o $(CXXFLAGS) $(DEFS) 

summa.o: summa.cxx topo_pdgemm_algs.h
	$(CXX) -c summa.cxx -o summa.o $(CXXFLAGS) $(DEFS) 

dual_cannon.o: dual_cannon.cxx topo_pdgemm_algs.h
	$(CXX) -c dual_cannon.cxx -o dual_cannon.o $(CXXFLAGS) $(DEFS) 


clean:
	rm -f *.o 
