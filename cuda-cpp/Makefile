CU_APPS=coalescing finite-difference optimize-data-transfers-bandwidthtest optimize-data-transfers-profile overlap-data-transfers-async shared-memory transpose
C_APPS=

all: ${C_APPS} ${CU_APPS}

%: %.cu
	nvcc -O2 -gencode=arch=compute_52,code=sm_52 -gencode=arch=compute_61,code=sm_61 -o $@ $< -lcudadevrt --ptxas-options=-v --relocatable-device-code true
%: %.c
	gcc -O2 -std=c99 -o $@ $<
clean:
	rm -f ${CU_APPS} ${C_APPS}
