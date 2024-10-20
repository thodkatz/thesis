import thesis.scbutterfly as scbutterfly
from thesis.preprocessing import preprocess_nault_no_filtering
from thesis.datasets import get_nault_multi_dose

if __name__ == "__main__":
    dataset = get_nault_multi_dose()
    scbutterfly.run_nault_all_dosages(name="no_filtering_nault", dataset=preprocess_nault_no_filtering(dataset))
    