CUDA_HOME?=$(shell dirname `dirname \`which nvcc\``)
CUDA_INC := ${CUDA_HOME}/include
NVCC := ${CUDA_HOME}/bin/nvcc
ARCH?=sm_37
CXXFLAGS:=-std=c++11 -O2 -D GOOGLE_CUDA=1
PYTHON?=$(shell which python)
TF_INC?=$(shell ${PYTHON} -c 'import tensorflow as tf; print(tf.sysconfig.get_include())')
TF_INCLUDE=-I ${TF_INC} -I ${TF_INC}/external/nsync/public
TF_LDFLAGS?=$(shell ${PYTHON} -c 'import tensorflow as tf; v=[int(a) for a in tf.__version__.split(".")[:2]]; new_tf=v[0]>1 or (v[0]==1 and v[1]>=4); print("-L "+tf.sysconfig.get_lib()+" -ltensorflow_framework" if new_tf else "")')
USE_OLD_EABI?=$(shell ${PYTHON} -c 'import tensorflow as tf; print((1-tf.__cxx11_abi_flag__) if hasattr(tf,"__cxx11_abi_flag__") else int(int(tf.__compiler_version__[0])<6))')
ifeq (${USE_OLD_EABI}, 1)
	CXXFLAGS += -D_GLIBCXX_USE_CXX11_ABI=0
endif
LIB=-L ${CUDA_HOME}/targets/x86_64-linux/lib -L ${CUDA_HOME}/lib64/ -L ${CUDA_HOME}/extras/CUPTI/lib64/ -lcudart ${TF_LDFLAGS}

all: roi_pooling.so

roi_pooling_grad.o: roi_pooling_grad.cc
	${CXX} ${CXXFLAGS} -c -o roi_pooling_grad.o roi_pooling_grad.cc ${TF_INCLUDE} -I ${CUDA_INC} -fPIC -O2

roi_pooling.o: roi_pooling.cc
	${CXX}  ${CXXFLAGS} -c -o roi_pooling.o roi_pooling.cc ${TF_INCLUDE} -I ${CUDA_INC} -fPIC -O2

roi_pooling.cu.o: roi_pooling.cu.cc
	${NVCC} -std=c++11 -c -o roi_pooling.cu.o roi_pooling.cu.cc ${TF_INCLUDE} -I ${CUDA_INC} -D GOOGLE_CUDA=1 -x cu -Xcompiler -fPIC -arch=${ARCH} -use_fast_math

roi_pooling_grad.cu.o: roi_pooling_grad.cu.cc
	${NVCC} -std=c++11 -c -o roi_pooling_grad.cu.o roi_pooling_grad.cu.cc ${TF_INCLUDE} -I ${CUDA_INC} -D GOOGLE_CUDA=1 -x cu -Xcompiler -fPIC -arch=${ARCH} -use_fast_math


roi_pooling.so: roi_pooling.o roi_pooling_grad.o roi_pooling.cu.o roi_pooling_grad.cu.o
	${CXX}  ${CXXFLAGS} -shared -o roi_pooling.so roi_pooling.o roi_pooling_grad.o roi_pooling.cu.o roi_pooling_grad.cu.o ${TF_INCLUDE} -I ${CUDA_INC} -fPIC -O2 ${LIB}


clean:
	-rm *.so
	-rm *.o