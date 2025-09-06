from schedulefree import RAdamScheduleFree

from src.modules.data_process.aggregator import (
    Aggregator,
    AttentionPoolingAggregator,
    EndsSegmentMeanAggregator,
    LogSumExpAggregator,
    WeightedMeanAggregator,
)
from src.modules.data_process.data_process_list import DataProcessList
from src.modules.data_process.initializer import Initializer
from src.modules.data_process.positional_encoder import (
    BidirectionalLearnableRoPEPositionalEncoder,
    LearnableAbsolutePositionalAdder,
    LearnableAbsolutePositionalScaler,
    LearnableFourierPositionalEncoder,
    LearnableRoPEPositionalEncoder,
    FloaterPositionalEncoder,
    RoPEPositionalEncoder,
    SinusoidalPositionalEncoder,
)
from src.modules.data_process.positional_encoder.floater_dynamics import (
    SimpleGatedSineDynamics,
)
from src.modules.dataloader.dataloader import Dataloader, DataloaderConfig
from src.modules.model.basic import BasicModel
from src.modules.protein.protein_list import ProteinList
from src.modules.train.trainer import Trainer

# seed = 5900308802214385025 is better
# seed = 5911646470734835975 is more better
# seed = 10338668691691671231 is more and more better
# seed = 5138387061145830358 is bad

protein_list = ProteinList.load_from_hdf5("outputs/plasma_lumos_1h/plasma_lumos_1h_data_esm2.h5").shuffle(
    # seed=10338668691691671231
)

input_props = []
output_props = ["rt"]

initializer = Initializer()

dim = 1280

# 既存のエンコーダ例（未使用だが残して参考にする場合はコメントアウト）
# sipe = SinusoidalPositionalEncoder(a=10000, b=1, gamma=1 / 2)
# rope = RoPEPositionalEncoder(theta_base=10000)
# lfpe = LearnableFourierPositionalEncoder(64, 10.0, 10000.0, 0.1)
# lape = LearnableAbsolutePositionalAdder(max_length=30)
# lapes = LearnableAbsolutePositionalScaler(max_length=30)
# lrope = LearnableRoPEPositionalEncoder(dim=dim, theta_base=10000)
# blrope = BidirectionalLearnableRoPEPositionalEncoder(dim=dim, theta_base=10000)

# FLOATER 用 Dynamics とエンコーダ
dynamics = SimpleGatedSineDynamics(
    dim=dim, hidden=256, num_freqs=8, omega_min=0.1, omega_max=10.0, damping=0.1
)
floater = FloaterPositionalEncoder(
    dim=dim,
    dynamics=dynamics,
    delta_t=0.5,
    method="dopri5",
    rtol=1e-3,
    atol=1e-6,
    use_adjoint=True,
)

agg = Aggregator("mean")
wagg = WeightedMeanAggregator(dim=1280, max_length=64)
esmagg = EndsSegmentMeanAggregator(head_len=8, tail_len=8)
lsagg = LogSumExpAggregator(tau=5.0)
apagg = AttentionPoolingAggregator(dim=1280, num_queries=2, temperature=5.0)

process_list = DataProcessList(iterable=[initializer, floater, agg])

dataloader_config = DataloaderConfig(
    protein_list=protein_list,
    input_props=input_props,
    output_props=output_props,
    batch_size=32,
    cacheable=False,
    process_list=process_list,
)
dataloader = Dataloader(config=dataloader_config)

model = BasicModel(input_dim=dataloader.output_dim(input_dim=1280))

optimizer = RAdamScheduleFree(
    [
        {"params": model.parameters(), "lr": 1e-3},
        {"params": floater.parameters(), "lr": 5e-4},
        # {"params": apagg.parameters(), "lr": 5e-4},
    ]
)

trainer = Trainer(model=model, dataloader=dataloader, optimizer=optimizer)

trainer.train()

acc = trainer._recorder.belle_epoch_summary.evaluate.accuracy

print(acc)
