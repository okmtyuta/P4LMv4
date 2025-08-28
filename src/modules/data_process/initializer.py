from src.modules.data_process.data_process import DataProcess
from src.modules.protein.protein import Protein


class Initializer(DataProcess):
    @property
    def dim_factor(self) -> int:
        return 1

    def _act(self, protein: Protein):
        reps = protein.get_representations()
        return protein.set_processed(processed=reps)
