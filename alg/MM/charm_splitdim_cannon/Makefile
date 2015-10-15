CHARM_HOME=/home/edgar/work/charm
OPTS=-g -O0
CHARMC=$(CHARM_HOME)/bin/charmc $(OPTS)

MATH_LIB        = -lesslbg -lesslsmpbg -lxlf90_r  \
		  -lmass -lmassv -lxlfmath -lxlomp_ser -lxlsmp -lpthread

LOCAL_LIBS      = -L/soft/apps/LAPACK \
		  -L/opt/ibmcmp/xlf/bg/11.1/bglib \
		  -L/opt/ibmcmp/xlsmp/bg/1.7/bglib \
		  -L/bgsys/ibm_essl/sles10/prod/opt/ibmmath/lib \
		  -L/bgsys/drivers/ppcfloor/gnu-linux/powerpc-bgp-linux/lib \
		  -L/gpfs/home/bohm/zlib/lib

#LIBS		= $(MATH_LIB) $(LOCAL_LIBS)
LIBS		= -lblas -llapack

#INCLUDE	= -I/bgsys/ibm_essl/sles10/prod/opt/ibmmath/include


OBJS = vspcannon.o

all: bench_vspc test_vspc

bench_vspc: $(OBJS) bench_vspc.o
	$(CHARMC) -language charm++ $(LIBS) -o bench_vspc $(OBJS) bench_vspc.o

test_vspc: $(OBJS) test_vspc.o
	$(CHARMC) -language charm++ $(LIBS) -o test_vspc $(OBJS) test_vspc.o

projections: $(OBJS) bench_vspc.o
	$(CHARMC) -language charm++ $(LIBS) -tracemode projections -lz -o bench_vspc.prj $(OBJS) bench_vspc.o

summary: $(OBJS) bench_vspc.o
	$(CHARMC) -language charm++ $(LIBS) -tracemode summary -lz -o bench_vspc.sum $(OBJS) bench_vspc.o

vspcannon.decl.h: vspcannon.ci
	$(CHARMC)  vspcannon.ci


bench_vspc.o: run_vspc.cxx vspcannon.h vspcannon.decl.h spcannon_internal.h
	$(CHARMC) -c $(INCLUDE) run_vspc.cxx -o bench_vspc.o -DBENCH

test_vspc.o: run_vspc.cxx vspcannon.h vspcannon.decl.h spcannon_internal.h
	$(CHARMC) -c $(INCLUDE) run_vspc.cxx -o test_vspc.o -DTEST

vspcannon.o: vspcannon.cxx vspcannon.h vspcannon.decl.h spcannon_internal.h
	$(CHARMC) -c $(INCLUDE) vspcannon.cxx

clean:
	rm -f *.decl.h *.def.h conv-host *.o bench_vspc test_vspc bench_vspc.prj charmrun *~

