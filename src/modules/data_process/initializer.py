from src.modules.data_process.data_process import DataProcess
from src.modules.protein.protein import Protein


class Initializer(DataProcess):
    def _act(self, protein: Protein):
        return protein.set_processed(processed=protein.representations)
