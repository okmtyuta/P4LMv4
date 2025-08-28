from src.modules.data_process.rope_positional_encoder import RoPEPositionalEncoder
from src.modules.train.trainer import Trainer
from src.modules.model.basic import BasicModel
from src.modules.dataloader.dataloader import Dataloader, DataloaderConfig
from src.modules.data_process.data_process_list import DataProcessList
from src.modules.data_process.initializer import Initializer
from src.modules.data_process.aggregator import Aggregator
from src.main.extraction import ExtractionRunner
from src.modules.protein.protein_list import ProteinList
from src.main.configs.extraction.esm2.plasma_lumos_1h import plasma_lumos_1h_config

# seed = 5900308802214385025 is better
# seed = 5911646470734835975 is more better
# seed = 10338668691691671231 is more and more better
protein_list = ProteinList.load_from_hdf5("outputs/plasma_lumos_1h/plasma_lumos_1h_data_esm2.h5").shuffle(seed=10338668691691671231)

input_props = ["length"]
output_props = ["rt"]

initializer = Initializer()
positional_encoder = RoPEPositionalEncoder(theta_base=10000)
aggregator = Aggregator("mean")

process_list = DataProcessList(iterable=[initializer, aggregator])

dataloader_config = DataloaderConfig(
    protein_list=protein_list,
    input_props=input_props,
    output_props=output_props,
    batch_size=32,
    cacheable=True,
    process_list=process_list,
)
dataloader = Dataloader(config=dataloader_config)


model = BasicModel(input_dim=1280 + 1)

trainer = Trainer(model=model, dataloader=dataloader)

trainer.train()
