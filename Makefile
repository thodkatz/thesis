SLURM_SCRIPTS = ./slurm_scripts
HPC_ALL_SCRIPT = ./scripts/all.sh

hpc:
	$(HPC_ALL_SCRIPT)

clean:
	rm -rf $(SLURM_SCRIPTS)

.PHONY: *