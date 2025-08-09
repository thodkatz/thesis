SLURM_SCRIPTS = ./slurm_scripts
HPC_ALL_SCRIPT = ./scripts/all.sh
HPARAM_SCRIPT = ./scripts/hparam.sh

hparam:
	$(HPARAM_SCRIPT)

hpc:
	$(HPC_ALL_SCRIPT)

clean:
	rm -rf $(SLURM_SCRIPTS)

setup_env:
	pip install -e .
	pip install -e ./lib/scgen
	pip install -e ./lib/scButterfly
	pip install -e ./lib/UnitedNet
	pip install -e ./lib/scPreGAN
	pip install -e ./lib/scVIDR
	mkdir -p data
	mkdir -p saved_results

.PHONY: *