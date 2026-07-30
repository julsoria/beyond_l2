.PHONY: install train-% explain-tesnet-cosine explain-pipnet-simplex explain-gaussian-isotropic-log clean-smoke

install:
	pip install -e .

# Usage: make train-% for % in one of:
#   protopnet-flowers102 protopnet-oxford_iiit_pet protopnet-cub200
#   tesnet-flowers102 tesnet-oxford_iiit_pet tesnet-cub200
#   pipnet-flowers102 pipnet-oxford_iiit_pet pipnet-cub200
#   protopool-cub200
#   protopnet_gaussian_iso-flowers102 protopnet_gaussian_iso-oxford_iiit_pet protopnet_gaussian_iso-cub200
train-%:
	$(eval ARCH := $(word 1,$(subst -, ,$*)))
	$(eval DATASET := $(word 2,$(subst -, ,$*)))
	cabrnet train --device cuda:0 --seed 42 \
		--config-dir configs/$(ARCH)/$(DATASET) \
		--output-dir runs/$(ARCH)_$(DATASET)

# A handful of representative explanation targets (one per paradigm family), each
# against a Flowers-102 checkpoint at runs/<arch>_flowers102. Adjust --data/--config to
# run other architecture/dataset/paradigm combinations directly via run_formal_exp.py.
explain-tesnet-cosine:
	python explain/run_formal_exp.py --paradigm cosine --data flowers102 --arch tesnet \
		--config runs/tesnet_flowers102

explain-pipnet-simplex:
	python explain/run_formal_exp.py --paradigm simplex --data flowers102 --arch pipnet \
		--config runs/pipnet_flowers102

explain-gaussian-isotropic-log:
	python explain/run_formal_exp.py --paradigm isotropic_log --data flowers102 --arch protopnet_gaussian_iso \
		--config runs/protopnet_gaussian_iso_flowers102

clean-smoke:
	rm -rf .smoketest
