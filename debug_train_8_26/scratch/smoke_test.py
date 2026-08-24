import sys, time
sys.path.insert(0, "/data/keeling/a/hytang2/Shallow_Water_System_On_Sphere")
import torch
from src.neural_operator.dataset import SWEDataset
from src.neural_operator.sfno_model import SphericalFourierNeuralOperator as SFNO
from src.neural_operator.loss import LOSS_FUNCTIONS

t0 = time.perf_counter()
training_data_dir = "model_output/numerical/resol_64/tau_(30000,2,2)/grid_eq/method_implicit/radiation_no_rad/duration_20/ic_rw/pressure_500/dataset_1980_2025_odd_month_500/"
ds = SWEDataset(simulation_data_dir=training_data_dir, n_future=48)
print(f"dataset loaded in {time.perf_counter()-t0:.1f}s, len={len(ds)}, nlat/nlon={ds.solver.nlat}/{ds.solver.nlon}")

device = torch.device("cuda")
model = SFNO(
    img_size=(ds.solver.nlat, ds.solver.nlon), grid=ds.solver.grid,
    num_layers=4, scale_factor=3, embed_dim=128,
    residual_prediction=False, inner_skip="linear",
    pos_embed="none", use_mlp=True, normalization_layer="layer_norm",
).to(device)
print(f"model built, n_params={sum(p.numel() for p in model.parameters())}")

u_curr, target = ds[0][0], ds[0][1]
u_curr = u_curr.unsqueeze(0).to(device)
target = target.unsqueeze(0).to(device)
print("u_curr shape", u_curr.shape, "target shape", target.shape)

opt = torch.optim.AdamW(model.parameters(), lr=8e-4, weight_decay=0.05)
solver = ds.solver
loss_fn = LOSS_FUNCTIONS["grid"]

pred = model(u_curr)
loss = loss_fn(solver, pred, target, relative=True)
loss.backward()
opt.step()
print(f"forward+backward OK, loss={loss.item():.6e}, total time={time.perf_counter()-t0:.1f}s")
