CU_APPS=globalVariable memTransfer pinMemTransfer readSegment \
		readSegmentUnroll sumArrayZerocpy \
		sumMatrixGPUManaged sumMatrixGPUManual transpose writeSegment
C_APPS=

all: ${C_APPS} ${CU_APPS}

%: %.cu
	nvcc -O2 -gencode=arch=compute_52,code=sm_52 -gencode=arch=compute_61,code=sm_61 -o $@ $<
%: %.c
	gcc -O2 -std=c99 -o $@ $<
clean:
	rm -f ${CU_APPS} ${C_APPS}
