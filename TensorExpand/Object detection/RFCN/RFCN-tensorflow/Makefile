all:
	$(MAKE) -C BoxEngine/ROIPooling all
	$(MAKE) -C Dataset/coco all

clean:
	$(MAKE) -C BoxEngine/ROIPooling clean
	$(MAKE) -C Dataset/coco clean
	rm -rf `find -iname __pycache__`
