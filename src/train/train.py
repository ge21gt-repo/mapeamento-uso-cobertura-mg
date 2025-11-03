import os
from src.train.evaluate import eval
import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torch import nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm
from torchvision import models
from transformers import SegformerForSemanticSegmentation, SegformerFeatureExtractor, Trainer, TrainingArguments
from utils.metrics import eval_mean_iou
from src.data.dataset import RemoteSensingDataset, load_pairs_torch
import time
import segmentation_models_pytorch as smp
import os

import matplotlib.pyplot as plt
import warnings
import mlflow
from utils.symlink_model import create_symlink

warnings.filterwarnings("ignore")


class SegmentationModel:
    def __init__(self, model_type, image_dir, mask_dir, num_classes=16, batch_size=16, num_epochs=160, patience=100,
                 learning_rate=6e-5, output_dir="./output", img_size=128, num_channels=12, device=None, save_steps=500):
        self.model_type = model_type.lower()
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.num_classes = num_classes
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.learning_rate = learning_rate
        self.output_dir = output_dir
        self.img_size = img_size
        self.save_steps = save_steps
        self.patience = patience
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.data_augmentation = False
        self.model = None
        self.feature_extractor = None
        self.num_channels = 12


    def create_model(self, model_name="nvidia/mit-b1"):
        if self.model_type == "deeplab":
            self.model = models.segmentation.deeplabv3_resnet50(weights=None)
            self.model.backbone.conv1 = nn.Conv2d(
                self.num_channels, 64, kernel_size=7, stride=2, padding=3, bias=False
            )
            self.model.classifier[4] = nn.Conv2d(256, self.num_classes, kernel_size=1)
            if self.model.aux_classifier is not None:
                self.model.aux_classifier[4] = nn.Conv2d(256, self.num_classes, kernel_size=1)
            
            self.model.to(self.device)

        elif self.model_type == "segformer":
            self.feature_extractor = SegformerFeatureExtractor(
                do_random_flip=True,
                do_resize=False,
                do_normalize=False,
                do_reduce_labels=False,
                size=self.img_size
            )
            self.model = SegformerForSemanticSegmentation.from_pretrained(
                model_name,
                num_labels=self.num_classes,
                ignore_mismatched_sizes=True,
                reshape_last_stage=True
            )
            self.model.to(self.device)

        elif self.model_type == "unet":
            self.model = smp.Unet(
                encoder_name=model_name,        # backbone (há muitos outros)
                encoder_weights=None,           
                in_channels=self.num_channels,  
                classes=self.num_classes        
            ).to(self.device)
        
        else:
            raise ValueError("Modelo desconhecido. Use 'segformer', 'deeplab' ou 'unet'.")


    def load_latest_weights(self):
        """
        Carrega automaticamente o último modelo salvo (via symlink *_latest).
        - DeepLab → carrega best_model.pth
        - UNet → carrega best_model.pth
        - SegFormer → carrega pasta final_model/
        """
        latest_dir = os.path.join(self.output_dir, f"{self.model_type}_latest")
        print(f"🔍 Procurando modelo mais recente em: {latest_dir}")

        # ============ SEGFORMER ============
        if self.model_type == "segformer":
            final_model_dir = os.path.join(latest_dir, "final_model")
            if os.path.exists(final_model_dir):
                print(f"🔁 Carregando modelo SegFormer de {final_model_dir}")
                self.feature_extractor = SegformerFeatureExtractor.from_pretrained(final_model_dir)
                self.model = SegformerForSemanticSegmentation.from_pretrained(final_model_dir)
                self.model.to(self.device)
                print("✅ Pesos SegFormer carregados com sucesso!")
            else:
                print("⚠️ Nenhum modelo SegFormer anterior encontrado. Criando novo modelo.")
                self.create_model()

        # ============ DEEPLAB ============
        elif self.model_type == "deeplab":
            model_path = os.path.join(latest_dir, "best_model.pth")
            if os.path.exists(model_path):
                print(f"🔁 Carregando pesos DeepLab de {model_path}")
                self.create_model()
                state_dict = torch.load(model_path, map_location=self.device)
                self.model.load_state_dict(state_dict, strict=False)
                print("✅ Pesos DeepLab carregados com sucesso!")
            else:
                print("⚠️ Nenhum modelo DeepLab anterior encontrado. Criando novo modelo.")
                self.create_model()

        # ============ UNET ============
        elif self.model_type == "unet":
            model_path = os.path.join(latest_dir, "best_model.pth")
            if os.path.exists(model_path):
                print(f"🔁 Carregando pesos Unet de {model_path}")
                self.create_model(model_name="mit_b0")
                state_dict = torch.load(model_path, map_location=self.device)
                self.model.load_state_dict(state_dict, strict=False)
                print("✅ Pesos Unet carregados com sucesso!")
            else:
                print("⚠️ Nenhum modelo unet anterior encontrado. Criando novo modelo.")
                self.create_model(model_name="mit_b0")


    def train_epoch(self, loader, criterion, optimizer):
        self.model.train()
        running_loss = 0.0
        for images, masks in tqdm(loader, desc="Treinando"):
            images, masks = images.to(self.device), masks.to(self.device)
            outputs = self.model(images)
            main_output = outputs['out'] if isinstance(outputs, dict) else outputs
            loss = criterion(main_output, masks)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * images.size(0)
        return running_loss / len(loader.dataset)


    def validate_epoch(self, loader, criterion):
        self.model.eval()
        running_loss = 0.0
        with torch.no_grad():
            for images, masks in tqdm(loader, desc="Validando"):
                images, masks = images.to(self.device), masks.to(self.device)
                outputs = self.model(images)
                main_output = outputs['out'] if isinstance(outputs, dict) else outputs
                loss = criterion(main_output, masks)
                running_loss += loss.item() * images.size(0)
        return running_loss / len(loader.dataset)


    def train(self):
        # ====== Identificação automática da versão ======
        model_base_name = f"{self.model_type}"
        version = 1
        while os.path.exists(os.path.join(self.output_dir, f"{model_base_name}_v{version}")):
            version += 1

        self.version_name = f"{model_base_name}_v{version}"
        self.version_dir = os.path.join(self.output_dir, self.version_name)
        os.makedirs(self.version_dir, exist_ok=True)

        print(f"🚀 Iniciando treinamento {self.version_name}")
        registered_model_name = f"{self.model_type.capitalize()}Model"

        train_img, train_mask, val_img, val_mask = load_pairs_torch(self.image_dir, self.mask_dir)

        if self.model_type == "segformer":
            mode = "multiband"
            feature_extractor = self.feature_extractor
            self.num_channels = 12
        else:
            mode = "rgb"
            feature_extractor = None

        train_dataset = RemoteSensingDataset(train_img, train_mask, feature_extractor, self.num_classes, mode)
        val_dataset = RemoteSensingDataset(val_img, val_mask, feature_extractor, self.num_classes, mode)

       
        # 🔹 Início do tracking MLflow
        if mlflow.active_run():
            print("Encerrando...")
            mlflow.end_run()

        with mlflow.start_run(run_name=f"{self.version_name}_training"):
            mlflow.log_params({
                "version": version,
                "model_type": self.model_type,
                "num_classes": self.num_classes,
                "batch_size": self.batch_size,
                "num_epochs": self.num_epochs,
                "learning_rate": self.learning_rate,
                "img_size": self.img_size,
                "num_channels": self.num_channels,
                "output_dir": self.version_dir
            })

            if self.model_type == "deeplab":
                train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True,
                                num_workers=4, pin_memory=True, drop_last=True)
                val_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False,
                                        num_workers=4, drop_last=False)
                
                criterion = nn.CrossEntropyLoss(ignore_index=-1)
                optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate, weight_decay=1e-4)
                scheduler = ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.5)

                train_history, val_history = [], []
                best_val_loss = float('inf')
                start_time = time.time()

                for epoch in range(self.num_epochs):
                    print(f"\nÉpoca {epoch+1}/{self.num_epochs}")
                    train_loss = self.train_epoch(train_loader, criterion, optimizer)
                    val_loss = self.validate_epoch(val_loader, criterion)
                    scheduler.step(val_loss)

                    train_history.append(train_loss)
                    val_history.append(val_loss)

                    # 🔹 Métricas MLflow
                    mlflow.log_metric("train_loss", train_loss, step=epoch)
                    mlflow.log_metric("val_loss", val_loss, step=epoch)
                    mlflow.log_metric("lr", optimizer.param_groups[0]['lr'], step=epoch)

                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        best_model_path = os.path.join(self.version_dir, "best_model.pth")
                        torch.save(self.model.state_dict(), best_model_path)
                        mlflow.log_artifact(best_model_path)
                        print(f"Melhor modelo salvo! Val_loss: {val_loss:.4f}")

                    if (epoch + 1) % 10 == 0:
                        epoch_model_path = os.path.join(self.version_dir, f"model_epoch_{epoch+1}.pth")
                        torch.save(self.model.state_dict(), epoch_model_path)
                        mlflow.log_artifact(epoch_model_path)

                    print(f"Treino: {train_loss:.4f} | Validação: {val_loss:.4f} | LR: {optimizer.param_groups[0]['lr']:.6f}")
                    
                    print(train_history)
                    print(val_history)
                
                # Curvas
                plt.figure(figsize=(10, 5))
                plt.plot(train_history, label='Treino')
                plt.plot(val_history, label='Validação')
                plt.title('Perda por Época')
                plt.xlabel('Época')
                plt.ylabel('Perda')
                plt.legend()
                loss_plot_path = os.path.join(self.version_dir, 'loss_curve.png')
                plt.savefig(loss_plot_path)
                mlflow.log_artifact(loss_plot_path)

                mlflow.log_metric("best_val_loss", best_val_loss)
                mlflow.log_metric("training_time_min", (time.time() - start_time) / 60)
                
                # evaluate
                evaluate = eval(self.model_type)
                mlflow.log_metric("Images validates", evaluate["dataset_size"])
                mlflow.log_metric("Pixel acc", evaluate["pixel_acc_mean"])
                mlflow.log_metric("Dice Macro", evaluate["dice_macro_mean"])
                mlflow.log_metric("Mean IOU", evaluate["mean_iou"])

        
            elif self.model_type == "segformer":
                training_args = TrainingArguments(
                    output_dir=self.version_dir,
                    learning_rate=self.learning_rate,
                    num_train_epochs=self.num_epochs,
                    per_device_train_batch_size=self.batch_size,
                    per_device_eval_batch_size=self.batch_size,
                    save_total_limit=3,
                    eval_strategy="steps",
                    save_strategy="steps",
                    save_steps=self.save_steps,
                    logging_steps=100,
                    eval_steps=self.save_steps,
                    load_best_model_at_end=True,
                    metric_for_best_model="mean_iou",
                    greater_is_better=True,
                )

                trainer = Trainer(
                    model=self.model,
                    args=training_args,
                    train_dataset=train_dataset,
                    eval_dataset=val_dataset,
                    compute_metrics=eval_mean_iou,
                )

                print("Iniciando treinamento SegFormer...")
                trainer.train()

                # 🔹 Registro MLflow
                final_model_path = os.path.join(self.version_dir, "final_model")
                trainer.save_model(final_model_path)
                self.feature_extractor.save_pretrained(final_model_path)
                mlflow.log_artifact(final_model_path)

                # evaluate
                evaluate = eval(self.model_type)
                mlflow.log_metric("Images validates", evaluate["dataset_size"])
                mlflow.log_metric("Pixel acc", evaluate["pixel_acc_mean"])
                mlflow.log_metric("Dice Macro", evaluate["dice_macro_mean"])
                mlflow.log_metric("Mean IOU", evaluate["mean_iou"])


            elif self.model_type == "unet":
                train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True,
                                        num_workers=4, pin_memory=True, drop_last=True)
                val_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False,
                                        num_workers=4, drop_last=False)
                
                criterion = nn.CrossEntropyLoss(ignore_index=-1)
                optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate, weight_decay=1e-4)
                scheduler = ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.5)

                train_history, val_history = [], []
                best_val_loss = float('inf')
                start_time = time.time()

                for epoch in range(self.num_epochs):
                    print(f"\nÉpoca {epoch+1}/{self.num_epochs}")
                    train_loss = self.train_epoch(train_loader, criterion, optimizer)
                    val_loss = self.validate_epoch(val_loader, criterion)
                    scheduler.step(val_loss)

                    train_history.append(train_loss)
                    val_history.append(val_loss)
                
                    # 🔹 Métricas MLflow
                    mlflow.log_metric("train_loss", train_loss, step=epoch)
                    mlflow.log_metric("val_loss", val_loss, step=epoch)
                    mlflow.log_metric("lr", optimizer.param_groups[0]['lr'], step=epoch)

                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        best_model_path = os.path.join(self.version_dir, "best_model.pth")
                        torch.save(self.model.state_dict(), best_model_path)
                        mlflow.log_artifact(best_model_path)
                        print(f"Melhor modelo salvo! Val_loss: {val_loss:.4f}")

                    if (epoch + 1) % 10 == 0:
                        epoch_model_path = os.path.join(self.version_dir, f"model_epoch_{epoch+1}.pth")
                        torch.save(self.model.state_dict(), epoch_model_path)
                        mlflow.log_artifact(epoch_model_path)

                    print(f"Treino: {train_loss:.4f} | Validação: {val_loss:.4f} | LR: {optimizer.param_groups[0]['lr']:.6f}")

                # Curvas
                plt.figure(figsize=(10, 5))
                plt.plot(train_history, label='Treino')
                plt.plot(val_history, label='Validação')
                plt.title('Perda por Época')
                plt.xlabel('Época')
                plt.ylabel('Perda')
                plt.legend()
                loss_plot_path = os.path.join(self.version_dir, 'loss_curve.png')
                plt.savefig(loss_plot_path)
                mlflow.log_artifact(loss_plot_path)

                mlflow.log_metric("best_val_loss", best_val_loss)
                mlflow.log_metric("training_time_min", (time.time() - start_time) / 60)
                
                # evaluate
                evaluate = eval(self.model_type)
                mlflow.log_metric("Images validates", evaluate["dataset_size"])
                mlflow.log_metric("Pixel acc", evaluate["pixel_acc_mean"])
                mlflow.log_metric("Dice Macro", evaluate["dice_macro_mean"])
                mlflow.log_metric("Mean IOU", evaluate["mean_iou"])
                
            
            out = f"{self.output_dir}/{self.model_type}_latest"
            create_symlink(self.version_dir, out)

            # 🔹 Finaliza o run MLflow
            print(f"\n✅ Treinamento concluído: {self.version_name}")
            print(f"📁 Resultados salvos em: {self.version_dir}")
            print(f"📊 Run registrada no MLflow como: {self.version_name}_training")

            try:
                if self.model_type == "unet":
                    mlflow.pytorch.log_model(self.model, artifact_path="model", registered_model_name=registered_model_name)
                elif self.model_type == "deeplab":
                    mlflow.pytorch.log_model(self.model, artifact_path="model", registered_model_name=registered_model_name)
                elif self.model_type == "segformer":
                    mlflow.pytorch.log_model(self.model, artifact_path="model", registered_model_name=registered_model_name)
                print(f"✅ Modelo registrado como '{registered_model_name}' no MLflow!")
            except Exception as e:
                print(f"⚠️ Erro ao registrar modelo no MLflow: {e}")
           
            print("Encerrando...")
            mlflow.end_run()


def segformer(use_weights=False, **kwargs):
    kwargs['model_type'] = 'segformer'
    model = SegmentationModel(**kwargs)
    if use_weights:
        model.load_latest_weights()
    else:
        model.create_model(model_name="nvidia/mit-b1")
    return model


def deeplab(use_weights=False, **kwargs):
    kwargs['model_type'] = 'deeplab'
    model = SegmentationModel(**kwargs)
    if use_weights:
        model.load_latest_weights()
    else:
        model.create_model()
    return model


def unet(use_weights=False, **kwargs):
    kwargs['model_type'] = 'unet'
    model = SegmentationModel(**kwargs)
    if use_weights:
        model.load_latest_weights()
    else:
        model.create_model(model_name="mit_b0")
    return model
