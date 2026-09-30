red:=$(shell tput bold ; tput setaf 1)
green:=$(shell tput bold ; tput setaf 2)
yellow:=$(shell tput bold ; tput setaf 3)
blue:=$(shell tput bold ; tput setaf 4)
magenta:=$(shell tput bold ; tput setaf 5)
cyan:=$(shell tput bold ; tput setaf 6)
reset:=$(shell tput sgr0)


data/TOY:
	python gen_toy.py --dest $@ -n 10 10 -wh 256 256 -r 50

data/TOY2:
	rm -rf $@_tmp $@
	python examples/gen_two_circles.py --dest $@_tmp -n 1000 100 -r 25 -wh 256 256
	mv $@_tmp $@


# Extraction and slicing for Segthor
## Original one
data/segthor_part1: data/segthor_part1.zip
	$(info $(yellow)unzip $<$(reset))
	sha256sum -c data/segthor_part1.sha256
	unzip -q $<
	rm -f $@/.DS_STORE

data/segthor_train_full: data/segthor_train_full.zip
	$(info $(yellow)unzip $<$(reset))
	sha256sum -c data/segthor_train_full.sha256
	rm -rf $@_tmp
	mkdir -p $@_tmp
	unzip -q $< -d $@_tmp
	mv $@_tmp $@

data/SEGTHOR:
	$(info $(green)python $(CFLAGS) -m preprocessing.slice_segthor$(reset))
	rm -rf $@_tmp $@
	python $(CFLAGS) -m preprocessing.slice_segthor --source_dir data/segthor_part1 --dest_dir $@_tmp \
		--shape 256 256 --target-spacing 1.5 1.5 --window -1000 1000 --retains 5
	mv $@_tmp $@

## Full clean dataset: 30 training patients and 10 validation patients
data/SEGTHOR_FULL: data/segthor_train_full
	$(info $(green)python $(CFLAGS) -m preprocessing.slice_segthor$(reset))
	rm -rf $@_tmp $@
	python $(CFLAGS) -m preprocessing.slice_segthor --source_dir data/segthor_train_full --dest_dir $@_tmp \
		--shape 256 256 --target-spacing 1.5 1.5 --window -1000 1000 --retains 10 --seed 0
	mv $@_tmp $@

## Corrected ground truth
data/SEGTHOR_CORRECTED:
	$(info $(green)python $(CFLAGS) -m preprocessing.slice_segthor$(reset))
	find data/correct_data -name '.DS_Store' -delete
	rm -rf $@_tmp $@
	python $(CFLAGS) -m preprocessing.slice_segthor --source_dir data/correct_data --dest_dir $@_tmp \
		--shape 256 256 --target-spacing 1.5 1.5 --window -1000 1000 --retains 5
	mv $@_tmp $@
