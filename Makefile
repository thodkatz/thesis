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
	#conda env create -n $(ENV_NAME) -f environment.yml
	conda run -n $(ENV_NAME) pip install -e .
	conda run -n $(ENV_NAME) pip install -e ./lib/scgen
	conda run -n $(ENV_NAME) pip install -e ./lib/scButterfly
	conda run -n $(ENV_NAME) pip install -e ./lib/UnitedNet
	conda run -n $(ENV_NAME) pip install -e ./lib/scPreGAN
	conda run -n $(ENV_NAME) pip install -e ./lib/scVIDR
	mkdir -p data
	mkdir -p saved_results

.PHONY: *