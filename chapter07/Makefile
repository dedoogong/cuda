CU_APPS=debug-hazards debug-segfault \
        debug-segfault.fixed sumMatrixGPU sumMatrixGPU_nvToolsExt

all: ${CU_APPS}

sumMatrixGPU_nvToolsExt: sumMatrixGPU_nvToolsExt.cu
	nvcc -O2 -gencode=arch=compute_52,code=sm_52 -gencode=arch=compute_61,code=sm_61 -L${CUDA_HOME}/lib64 -lnvToolsExt -o sumMatrixGPU_nvToolsExt sumMatrixGPU_nvToolsExt.cu
%: %.cu
	nvcc -O2 -gencode=arch=compute_52,code=sm_52 -gencode=arch=compute_61,code=sm_61 -o $@ $<
%: %.c
	gcc -O2 -std=c99 -o $@ $<
clean:
	rm -f ${CU_APPS} ${C_APPS}
