HIPCC=/opt/rocm-3.3.0/hip/bin/hipcc -D_GLIBCXX_USE_CXX11_ABI=0

HIP_PATH ?= /opt/rocm-3.3.0/hip

MPI_LIB := -L/usr/mpi/gcc/openmpi-4.0.2rc3/lib -Xcompiler "-fPIC"
MPI_INC := -I/usr/mpi/gcc/openmpi-4.0.2rc3/include 
HIP_LIB := -L./ -L$(HIP_PATH)/lib64 -L/usr/lib -L/opt/rocm-3.3.0/lib -L/opt/rocm-3.3.0/hcc/lib -L/opt/rocm-3.3.0/hsa/lib -L/opt/rocm/rccl/lib
HIP_INC := -I./ -I$(HIP_PATH)/include -I/usr/include -I/opt/rocm-3.3.0/include  -I/opt/rocm-3.3.0/hcc/include -I/opt/rocm-3.3.0/hsa/include -I/opt/rocm/rccl/include #-D__HIP_PLATFORM_HCC__

CPP11 := -std=c++11 -D_GLIBCXX_USE_CXX11_ABI=0
CPPFLAGS := -std=c++11 $(HIP_INC) $(MPI_INC)
LDFLAGS  := $(HIP_LIB)  $(MPI_LIB) 
CFLAGS   += -O0
CXXFLAGS += -O0
CFLAGS   += -g
CXXFLAGS += -g
LIBS     := -libverbs -lrdmacm -lpthread -ldl -lm -lmpi -lhip_hcc -lrccl

all: test

test: test.cc 
	$(HIPCC) $(CXXFLAGS) $(CPPFLAGS) $(LDFLAGS) -o $@ $^ $(LIBS) 

clean:
	rm -f *.o $(EXES) lib*.{a,so}* *~ core.* *.so *.a test

.PHONY: driver clean all lib exes lib_install install
