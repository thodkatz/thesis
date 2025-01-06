SLURM_SCRIPTS = ./slurm_scripts
HPC_ALL_SCRIPT = ./scripts/all.sh
HPARAM_SCRIPT = ./scripts/hparam.sh

hparam:
	$(HPARAM_SCRIPT)

hpc:
	$(HPC_ALL_SCRIPT)

clean:
	rm -rf $(SLURM_SCRIPTS)

env:
	#conda env create -n thesis -f environment.yml
	conda run -n thesis pip install -e .
	conda run -n thesis pip install -e ./lib/scgen
	conda run -n thesis pip install -e ./lib/scButterfly
	conda run -n thesis pip install -e ./lib/UnitedNet
	conda run -n thesis pip install -e ./lib/scPreGAN
	conda run -n thesis pip install -e ./lib/scVIDR

.PHONY: *