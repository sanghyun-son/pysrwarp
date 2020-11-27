SUBDIRS = cuda

all: svf_cuda
	python setup.py install

svf_cuda:
	$(MAKE) -C cuda

clean:
	rm -rf build dist ./*.egg-info
	for dir in $(SUBDIRS); do $(MAKE) -C $$dir Makefile $@; done
