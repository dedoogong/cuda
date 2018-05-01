APPS=hello

all: ${APPS}

%: %.cu
	nvcc -O2 -gencode=arch=compute_52,code=sm_52 -gencode=arch=compute_61,code=sm_61 -o $@ $<
clean:
	rm -f ${APPS}
