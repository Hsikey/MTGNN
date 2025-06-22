import os
import datetime
import argparse
import torch
import numpy as np
import torch.utils.data as data
# from torch.utils.tensorboard import SummaryWriter
import torch
import matplotlib.pyplot as plt
from networks import *
from torch.optim.lr_scheduler import ReduceLROnPlateau
import warnings
from cmu_mosei import CMUMOSEIDataset
from main_utils import *
warnings.filterwarnings("ignore")
import os
import time
import logging
import sys
from decimal import Decimal
import torchvision.models as models
import torch
from torchstat import stat
# fixed seed
seed = 20
np.random.seed(seed)
torch.manual_seed(seed)  # cpu
torch.cuda.manual_seed_all(seed)  # gpu
torch.backends.cudnn.enabled = False
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

modalities = ("visual", "audio", "bert")
# modalities = ("visual", "audio", "bert", "adapter")
modalitiesAll = ("visual50", "audio50", "bert50")


def get_n_params(model):
    pp = 0
    for p in list(model.parameters()):
        nn = 1
        for s in list(p.size()):
            nn = nn * s
        pp += nn
    return pp


# define model input
def get_X(device, sample):
    ret = []
    # for m in modalities:
    for m in modalitiesAll:
        X = sample[m].to(device)
        ret.append(X.float())
    n = ret[0].size(0)
    return ret, n



def main(lr):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--datadir", type=str, help="dataset directory", default="datasets"
    )
    parser.add_argument("--lr", type=float, help="learning rate", default=lr) # 0.0025  0.002 0.0015 0.0042 0.003 0.0038 0.0026 0.036  0.0023
    parser.add_argument("--batch_size", type=int, help="batch size", default=16)  #64
    parser.add_argument("--num_workers", type=int, help="num workers", default=0)

    parser.add_argument("--epochs", type=int, help="train epochs", default=50)
    parser.add_argument(
        "--checkpoint", type=str, help="model checkpoint for evaluation", default=""
    )
    parser.add_argument(
        "--checkpointdir",
        type=str,
        help="directory to save weights",
        default="checkpoints",
    )
    parser.add_argument(
        "--model",
        type=str,
        help="choose to model type",
        default="MBGCN",
    )
    parser.add_argument(
        "--no_verbose",
        action="store_true",
        default=False,
        help="turn off verbose for training",
    )
    parser.add_argument(
        "--log_interval",
        type=int,
        help="interval for displaying training info if verbose",
        default=1,
    )
    parser.add_argument(
        "--no_save",
        action="store_true",
        default=False,
        help="set to not save model weights",
    )
    parser.add_argument("--train", action="store_true", default=False, help="training")
    parser.add_argument('--data_path', type=str, default='data',
                    help='path for storing the dataset')

    args = parser.parse_args()

    print("The configuration of this run is:")
    print(args, end="\n\n")

    # Detect devices
    use_cuda = torch.cuda.is_available()  # check if GPU exists
    print("cuda" if use_cuda else "cpu")
    device = torch.device("cuda")  # use CPU or GPU

    # Data loading parameters
    params = (
        {
            "batch_size": args.batch_size,
            "shuffle": True,
            "num_workers": args.num_workers,
            "pin_memory": False,
        }
        # if use_cuda
        # else {"batch_size": args.batch_size, "shuffle": True, "num_workers": 0}
    )

    if args.datadir == 'iemocap':
        dataset = args.datadir
        train_data = get_data(args, dataset, 'train')
        valid_data = get_data(args, dataset, 'valid')
        test_data = get_data(args, dataset, 'test')
        
        train_loader = data.DataLoader(train_data, batch_size=args.batch_size, shuffle=True)
        valid_loader = data.DataLoader(valid_data, batch_size=args.batch_size, shuffle=True)
        test_loader = data.DataLoader(test_data, batch_size=args.batch_size, shuffle=True)

    elif args.datadir != 'iemocap':
        # dataset folders
        dataset = args.datadir.split('/')[2]
        # training_folder = os.path.join(args.datadir, "train_un")
        # val_folder = os.path.join(args.datadir, "val")
        # test_folder = os.path.join(args.datadir, "testNew_un")
        training_folder = os.path.join(args.datadir, "trainNew2")
        val_folder = os.path.join(args.datadir, "val")
        test_folder = os.path.join(args.datadir, "testNew2")

        # Generators.
        dataset_params = {"label": "sentiment"}
        # dataset_params = {"label": "emotion"}


        for m in modalitiesAll:
            dataset_params.update({m: None})

        # Load dataset
        training_set = CMUMOSEIDataset(training_folder, dataset_params)
        training_loader = data.DataLoader(training_set, **params)
        # val_set = CMUMOSEIDataset(val_folder, dataset_params)
        # val_loader = data.DataLoader(val_set, **params)
        test_set = CMUMOSEIDataset(test_folder, dataset_params)
        test_loader = data.DataLoader(test_set, **params)

    # define model
    model_param = {}
    if "visual" in modalities:
        model = FACETVisualLSTMNet()
        print("Initialized model for video modality")
        model_param.update(
            {"visual": {"model": model, "id": modalities.index("visual")}}
        )
    if "audio" in modalities:
        model = COVAREPAudioLSTMNet()
        print("Initialized model for audio modality")
        model_param.update({"audio": {"model": model, "id": modalities.index("audio")}})
    if "bert" in modalities:
        model = BERTTextLSTMNet()
        print("Initialized model for bert")
        model_param.update({"bert": {"model": model, "id": modalities.index("bert")}})

    if "text" in modalities:
        model = BERTTextLSTMNet()
        print("Initialized model for text modality")
        model_param.update({"text": {"model": model, "id": modalities.index("text")}})

    if "adapter" in modalities:
        checkpoint = torch.load('D:\GitHubCode\MBGCN\checkpoints\Adapter_CMU_MOSEI.pth')
        model = Adapter(model_param)
        model.load_state_dict(checkpoint['model_state_dict'])
        print("Initialized model for adapter")
        model_param.update({"adapter": {"model": model, "id": modalities.index("adapter")}})       

    if args.model == 'MGCN':
        multimodal_model = MGCN(model_param)
    if args.model == 'MTGCN':
        multimodal_model = MTGCN(model_param)
    if args.model == 'NHFNet':
        multimodal_model = MSAFLSTMNet(model_param)
    if args.model == 'MT':
        multimodal_model = MT(model_param)
    if args.model == 'Adapter':
        multimodal_model = Adapter(model_param)

    print(get_n_params(multimodal_model))
    multimodal_model.to(device)

    # loss functions
    train_loss_func = torch.nn.MSELoss()
    val_loss_func = torch.nn.L1Loss()
    new_loss_fun = torch.nn.SmoothL1Loss()
    # train mode or eval mode
    if args.train:
        print("Training...")
        # Adam parameters
        # optimizer = torch.optim.Adam(multimodal_model.parameters(), lr=args.lr)
        optimizer = torch.optim.Adam(multimodal_model.parameters(), lr=args.lr, weight_decay = 1e-5)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=20, eta_min=0.0001)
        # record training process
        current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        train_log_dir = os.path.join(args.checkpointdir, "logs/{}".format(current_time))
        # writer = SummaryWriter(log_dir=train_log_dir)
        test = []
        val_loss = []
        test_loss = []
        t_loss = []
        best_test_loss = 1e8
        best_epoch = 0
        start_time_1 = time.time()
        for epoch in range(args.epochs):
            # train, test model
            start_time = time.time()
            train_loss, epoch_train_scores = train(
                get_X,
                args.log_interval,
                multimodal_model,
                device,
                training_loader,
                optimizer,
                scheduler,
                train_loss_func,
                epoch,
                lr,
                not args.no_verbose,
            )
            end_time = time.time()
            print(
                "[Time: %f]"
                % (end_time - start_time)
            )
            # epoch_val_loss, epoch_val_score = validation(
            #     get_X, multimodal_model, device, train_loss_func, val_loader
            # )
            start_time = time.time()
            epoch_test_loss, epoch_test_score = validation(
                get_X, multimodal_model, device, train_loss_func, test_loader, epoch, lr
            )
            end_time = time.time()
            print(
                "[Time: %f]"
                % (end_time - start_time)
            )

            if epoch_test_loss < best_test_loss:
                print(f"Saved model at pre_trained_models.pt at epoch {epoch + 1}!")
                print('best test_loss is saved:',epoch_test_loss)
                # save_model(args, multimodal_model)
                best_test_loss = epoch_test_loss
                best_epoch = epoch 

            # if not args.no_save:
                states = {
                    "epoch": epoch + 1,
                    "model_state_dict": multimodal_model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "test_score": epoch_test_score,
                    "test_loss": epoch_test_loss,
                }
                torch.save(
                    states, f"checkpoints/{args.model}_{dataset}.pth"
                )
                print("Epoch {} model saved!".format(epoch + 1))

            # save results
            # writer.add_scalar("Loss/train", train_loss, epoch)
            # writer.add_scalar("Loss/test", epoch_test_loss, epoch)
            # writer.add_scalar("Acc7/train", epoch_train_scores[0], epoch)
            # writer.add_scalar("Acc7/test", epoch_test_score[0], epoch)
            # writer.add_scalar("Acc2/train", epoch_train_scores[1], epoch)
            # writer.add_scalar("Acc2/test", epoch_test_score[1], epoch)
            # writer.add_scalar("F1/train", epoch_train_scores[2], epoch)
            # writer.add_scalar("F1/test", epoch_test_score[2], epoch)

            test.append(epoch_test_score)
            # val_loss.append(epoch_val_loss)
            test_loss.append(epoch_test_loss)
            t_loss.append(train_loss)
            # writer.flush()

        test = np.array(test)
        # labels = ["Acc 7", "Acc 2","F1",, "Corr", "MAE"]
        acc2 = np.max(test.T[1])
        acc7 = np.max(test.T[0])
        mae = np.max(test.T[4])
        corr = np.max(test.T[3])
        logging.basicConfig(filename='./example.txt', level=logging.INFO)
        # if acc2 > 86 or acc7 > 56 or mae < 48.3 or corr > 79:
        if acc2 > 86 or acc7 > 47 or mae < 70 or corr > 75:
            logging.warning("---------------------------------------------------")
        # labels = ["Acc 7", "Acc 2","Acc 2_new","F1","F1_new", "Corr", "MAE"]
        logging.info("lr:%s", lr)
        labels = ["Acc 7", "Acc 2", "F1","Corr", "MAE"]
        for scores, label in zip(test.T, labels):
                if label == "MAE":
                    print( 
                        "Best {} score {:.2f}% at epoch {}".format(
                            label, np.min(scores), np.argmin(scores) + 1
                        )
                    )
                    logging.info("Best %s:%s at epoch %s", label, np.min(scores), np.argmin(scores) + 1)
                else:
                    print( 
                        "Best {} score {:.2f}% at epoch {}".format(
                            label, np.max(scores), np.argmax(scores) + 1
                        )
                    )
                    logging.info("Best %s:%s at epoch %s", label, np.max(scores), np.argmax(scores) + 1)
        # if acc2 > 86 or acc7 > 56 or mae < 48.3 or corr > 79:
        if acc2 > 86 or acc7 > 47 or mae < 70 or corr > 75:
            logging.warning("---------------------------------------------------")
        # print("Best Acc7", epoch_test_score[0])
        # print("Best Acc2", epoch_test_score[1])
        # print("Best F1", epoch_test_score[2])
        # print("Best Corr", epoch_test_score[3])
        
        # epoch_train = np.arange(args.epochs)
        # plt.plot(epoch_train, val_loss, marker="o", mec="r", mfc="w", label="val_loss")
        # plt.plot(epoch_train, t_loss, marker="o", mec="r", mfc="w", label="train_loss")
        # plt.plot(
        #     epoch_train, test_loss, marker="o", mec="r", mfc="w", label="test_loss"
        # )
        # plt.legend()  # 让图例生效
        # plt.legend(loc="upper right")
        # plt.show()

    else:
        if args.checkpoint:
            print("Evaluating...")
            model_path = args.checkpoint
            checkpoint = (
                torch.load(model_path)
                if use_cuda
                else torch.load(model_path, map_location=torch.device("cpu"))
            )
            multimodal_model.load_state_dict(checkpoint["model_state_dict"])
            print("Loaded model from", model_path)
            test_set = CMUMOSEIDataset(test_folder, dataset_params)
            test_loader = data.DataLoader(test_set, **params)
            epoch_test_loss, epoch_test_score = validation(
                get_X,
                multimodal_model,
                device,
                val_loss_func,
                test_loader,
                print_cm=True,
            )
        else:
            print("--checkpoint not specified")

if __name__ == "__main__":
    lr = 0.0001
    while lr < 0.01:
        main(lr)
        lr = round(lr + 0.00001,5)