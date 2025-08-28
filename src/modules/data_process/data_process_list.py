from src.modules.container.sequence_container import SequenceContainer
from src.modules.data_process.data_process import DataProcess
from src.modules.protein.protein_list import ProteinList


class DataProcessList(SequenceContainer[DataProcess]):
    def __init__(self, iterable: list[DataProcess]):
        super().__init__(iterable=iterable)

    def __call__(self, protein_list: ProteinList) -> ProteinList:
        for process in self._data:
            protein_list = process(protein_list=protein_list)

        return protein_list
