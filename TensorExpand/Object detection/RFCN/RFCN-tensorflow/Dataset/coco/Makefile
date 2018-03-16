PYTHON?=$(shell which python)
all:
    # install pycocotools locally
	${PYTHON} setup.py build_ext --inplace
	rm -rf build

install:
	# install pycocotools to the Python site-packages
	${PYTHON} setup.py build_ext install
	rm -rf build

clean:
	rm -rf build 2>/dev/null
	rm -rf pycocotools/*.c 2>/dev/null
	rm -rf pycocotools/*.so 2>/dev/null
	rm -rf pycocotools/__pycache__ 2>/dev/null