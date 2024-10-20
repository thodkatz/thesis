import thesis.scbutterfly as scbutterfly
from thesis.datasets import get_pbmc

if __name__ == "__main__":
    dataset = get_pbmc()
    name = "pbmc"
    scbutterfly.run_pbmc(name=name, dataset=dataset)
