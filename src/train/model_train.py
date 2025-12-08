import os

import torch
import torch.nn as nn
from torch.optim import Adam, Optimizer
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
from torchmetrics.image.fid import FrechetInceptionDistance
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

from src._utils import _load_yaml
from src.data import OptiSARDataset
from src.model import Discriminator, Generator

torch.backends.cudnn.benchmark = True


def train_model(
    batch_size: int = 8,
    epochs: int = 10,
    real_label: float = 0.9,
    fake_label: float = 0.0,
    lpips_batch_size: int = 8,
    fid_batch_size: int = 32,
    num_fid_features: int = 64,
    num_fid_to_find: int = 156,
    lpips_epoch: int = 5,
    fid_epoch: int = 10,
    optimizer_G: Optimizer = Adam,
    optimizer_F: Optimizer = Adam,
    optimizer_D_X: Optimizer = Adam,
    optimizer_D_Y: Optimizer = Adam,
    adam_betas: tuple | None = (0.5, 0.999),
    lr_G: float = 2e-4,
    lr_F: float = 2e-4,
    lr_D_X: float = 2e-4,
    lr_D_Y: float = 2e-4,
    lambda_cyc: float = 10.0,
    # lambda_id: float = 3.0,
    rgb_img_shape: tuple = (3, 256, 256),
    generator_G_save_path: str = "./src/train/checkpoints/generator_G.pth",
    generator_F_save_path: str = "./src/train/checkpoints/generator_F.pth",
    discriminator_X_save_path: str = "./src/train/checkpoints/discriminator_X.pth",
    discriminator_Y_save_path: str = "./src/train/checkpoints/discriminator_Y.pth",
):
    use_amp: bool = False  # Weather to use autocast and Gradscaler or not
    if torch.cuda.is_available():
        use_amp = True  # Training is faster when autocast is not used. I don't know why
        torch.set_default_device("cuda")
    else:
        torch.set_default_device("cpu")
    device = torch.get_default_device()

    scaler = torch.amp.GradScaler(enabled=use_amp)

    print("--------------------------------------")
    print(f"Implementation working on device: {device}")
    print("--------------------------------------")

    dataset_info: dict = _load_yaml("./src/train/yaml/data_config.yaml")["data"]
    dataset = OptiSARDataset(
        root_dir=dataset_info["root_name"],
        train_sar=dataset_info["train_dir"][0],
        train_rgb=dataset_info["train_dir"][1],
    )

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=2,
        pin_memory=True,
        prefetch_factor=4,
        persistent_workers=True,
    )

    # We will set generator G for converting SAR -> RGB
    # and, generator F for converting RBG -> SAR
    # Discriminator X will test SAR images and the
    # Discriminator Y will test RGB images

    generator_G = Generator(img_channels_in=1, img_channels_out=3).to(device)
    generator_F = Generator(img_channels_in=3, img_channels_out=1).to(device)

    discriminator_X = Discriminator(in_channels=1).to(device)
    discriminator_Y = Discriminator(in_channels=3).to(device)

    # print("G:", next(generator_G.parameters()).device)
    # print("F:", next(generator_F.parameters()).device)
    # print("D_X:", next(discriminator_X.parameters()).device)
    # print("D_Y:", next(discriminator_Y.parameters()).device)

    if (
        optimizer_G is Adam
        or optimizer_F is Adam
        or optimizer_D_X is Adam
        or optimizer_D_Y is Adam
    ):
        if not adam_betas:
            adam_betas = (0.5, 0.999)

    if optimizer_G is Adam:
        optimizer_G = optimizer_G(generator_G.parameters(), lr=lr_G, betas=adam_betas)
    else:
        optimizer_G = optimizer_G(generator_G.parameters(), lr=lr_G)

    if optimizer_F is Adam:
        optimizer_F = optimizer_F(generator_F.parameters(), lr=lr_F, betas=adam_betas)
    else:
        optimizer_F = optimizer_F(generator_F.parameters(), lr=lr_F)

    if optimizer_D_X is Adam:
        optimizer_D_X = optimizer_D_X(
            discriminator_X.parameters(), lr=lr_D_X, betas=adam_betas
        )
    else:
        optimizer_D_X = optimizer_D_X(discriminator_X.parameters(), lr=lr_D_X)

    if optimizer_D_Y is Adam:
        optimizer_D_Y = optimizer_D_Y(
            discriminator_Y.parameters(), lr=lr_D_Y, betas=adam_betas
        )
    else:
        optimizer_D_Y = optimizer_D_Y(discriminator_Y.parameters(), lr=lr_D_Y)

    decay_epoch = epochs // 2

    def lr_rule(ep):
        if epochs == decay_epoch:
            return 1.0
        return (
            1.0
            if ep < decay_epoch
            else 1.0 - float(ep - decay_epoch) / float(epochs - decay_epoch)
        )

    scheduler_G = LambdaLR(optimizer_G, lr_lambda=lr_rule)
    scheduler_F = LambdaLR(optimizer_F, lr_lambda=lr_rule)
    scheduler_D_X = LambdaLR(optimizer_D_X, lr_lambda=lr_rule)
    scheduler_D_Y = LambdaLR(optimizer_D_Y, lr_lambda=lr_rule)

    mse = nn.MSELoss()
    l1 = nn.L1Loss()

    generator_G.train()
    generator_F.train()
    discriminator_X.train()
    discriminator_Y.train()

    lpips = LearnedPerceptualImagePatchSimilarity(net_type="alex").to(device)
    lpips_dataloader = DataLoader(
        dataset,
        batch_size=lpips_batch_size,
        num_workers=2,
        pin_memory=True,
        prefetch_factor=4,
        persistent_workers=True,
    )

    fid = FrechetInceptionDistance(
        feature=num_fid_features, input_img_size=rgb_img_shape, normalize=True
    ).to(device)
    fid_dataloader = DataLoader(
        dataset,
        batch_size=fid_batch_size,
        num_workers=2,
        pin_memory=True,
        prefetch_factor=4,
        persistent_workers=True,
    )

    for epoch in range(1, epochs + 1):
        print("----------------------------------")
        print(f"------- Epoch: [{epoch}/{epochs}] ----------")
        for dataloader_idx, bat in enumerate(dataloader):
            with torch.amp.autocast(device_type="cuda", enabled=use_amp):
                sar_tensor, rgb_tensor = bat  # Unpack the list
                sar_real = sar_tensor.to(device, non_blocking=True)
                rgb_real = rgb_tensor.to(device, non_blocking=True)

                # -------- Step 1 --------
                fake_rgb = generator_G(sar_real)
                fake_sar = generator_F(rgb_real)
                fake_rgb_detached = fake_rgb.detach()
                fake_sar_detached = fake_sar.detach()

                # -------- Step 2 --------
                rgb_pred_real = discriminator_Y(rgb_real)
                rgb_pred_fake = discriminator_Y(fake_rgb_detached)
                loss_D_Y = 0.5 * (
                    mse(rgb_pred_real, torch.full_like(rgb_pred_real, real_label))
                    + mse(rgb_pred_fake, torch.full_like(rgb_pred_fake, fake_label))
                )

            optimizer_D_Y.zero_grad()
            scaler.scale(loss_D_Y).backward()
            scaler.step(optimizer_D_Y)
            scaler.update()

            with torch.amp.autocast(device_type="cuda", enabled=use_amp):
                # -------- Step 3 --------
                sar_pred_real = discriminator_X(sar_real)
                sar_pred_fake = discriminator_X(fake_sar_detached)
                loss_D_X = 0.5 * (
                    mse(sar_pred_real, torch.full_like(sar_pred_real, real_label))
                    + mse(sar_pred_fake, torch.full_like(sar_pred_fake, fake_label))
                )

            optimizer_D_X.zero_grad()
            scaler.scale(loss_D_X).backward()
            scaler.step(optimizer_D_X)
            scaler.update()

            with torch.amp.autocast(device_type="cuda", enabled=use_amp):
                # -------- Step 4 --------
                discriminated_rgb_from_sar = discriminator_Y(fake_rgb)
                discriminated_sar_from_rgb = discriminator_X(fake_sar)

                L_adv_G = mse(
                    discriminated_rgb_from_sar,
                    torch.full_like(discriminated_rgb_from_sar, real_label),
                )
                L_adv_F = mse(
                    discriminated_sar_from_rgb,
                    torch.full_like(discriminated_sar_from_rgb, real_label),
                )

                # -------- Step 5 --------
                sar_reconstructed = generator_F(fake_rgb)
                rgb_reconstructed = generator_G(fake_sar)

                L_cyclic = l1(sar_reconstructed, sar_real) + l1(
                    rgb_reconstructed, rgb_real
                )

                # -------- Step 6 --------
                # identity_rgb = generator_G(rgb_real)
                # identity_sar = generator_F(sar_real)

                # L_identity = l1(identity_rgb, rgb_real) + l1(identity_sar, sar_real)

                # ------- Step 7 ---------
                L_generator = (
                    L_adv_G
                    + L_adv_F
                    + lambda_cyc * L_cyclic
                    # + lambda_id * L_identity
                )

            optimizer_G.zero_grad()
            optimizer_F.zero_grad()
            scaler.scale(L_generator).backward()
            scaler.step(optimizer_G)
            scaler.step(optimizer_F)
            scaler.update()

            if dataloader_idx % 100 == 0:
                print("Working...")

        scheduler_G.step()
        scheduler_F.step()
        scheduler_D_X.step()
        scheduler_D_Y.step()

        if epoch % lpips_epoch == 0:
            for _, batch_lpips in enumerate(lpips_dataloader):
                sar_img, rgb_img = batch_lpips  # Unpack the list properly
                generator_G.eval()
                with torch.no_grad():
                    sar_img = sar_img.to(device)
                    rgb_img = rgb_img.to(device)
                    generated_rgb_img = generator_G(sar_img)
                    lpips_score = lpips(generated_rgb_img, rgb_img)
                break
            generator_G.train()
            print(f"LPIPS score: {lpips_score.item()}")

        if epoch % fid_epoch == 0:
            with torch.no_grad():
                fid.reset()
                for id, batch_fid in enumerate(fid_dataloader):
                    sar_img, rgb_img = batch_fid  # Unpack the list properly
                    if id >= num_fid_to_find:
                        break
                    generator_G.eval()
                    sar_img = sar_img.to(device)
                    rgb_img = (
                        ((rgb_img.to(device) + 1.0) * 0.5)
                        .clamp(0.0, 1.0)
                        .to(dtype=torch.float32)
                    )  # Convert to [0, 1]
                    generated_rgb_img_fid = (
                        ((generator_G(sar_img) + 1.0) * 0.5)
                        .clamp(0.0, 1.0)
                        .to(dtype=torch.float32)
                    )  # convert to [0, 1]
                    fid.update(rgb_img, real=True)
                    fid.update(generated_rgb_img_fid, real=False)
                print(f"FID score: {fid.compute().item()}")
                fid.reset()
                generator_G.train()

        print("----------------------------------")

    print("--------- Training Finished -----------")
    print("Starting to save models...")

    try:
        os.makedirs(os.path.dirname(generator_G_save_path), exist_ok=True)
        torch.save(generator_G.state_dict(), generator_G_save_path)
        torch.save(generator_F.state_dict(), generator_F_save_path)
        torch.save(discriminator_X.state_dict(), discriminator_X_save_path)
        torch.save(discriminator_Y.state_dict(), discriminator_Y_save_path)
        print("All models saved successfully!")
    except Exception:
        print("Models failed to save.")


if __name__ == "__main__":
    train_model()
