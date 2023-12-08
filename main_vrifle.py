import sys, os
os.environ["CUDA_VISIBLE_DEVICES"]="0"
from deepspeech_pytorch.configs.inference_config import TranscribeConfig

import argparse
import wandb
from new_pytorch_deep_speech import PyTorchDeepSpeech
import numpy as np
from deepspeech_pytorch.model import DeepSpeech
from typing import TYPE_CHECKING, List, Optional, Tuple, Union
import torch, torchaudio
torch.set_num_threads(4)
from vrifle_attack import Attacker
from datetime import datetime

class DeepSpeech4Loss(PyTorchDeepSpeech):
    def __init__(
        self,
        model: Optional["DeepSpeech"] = None,
        pretrained_model: Optional[str] = None,
        device_type: str = "gpu",
        *args,
        **kwargs
    ):
        super().__init__(
            
            model=model,
            pretrained_model=pretrained_model,
            device_type=device_type,
            *args,
            **kwargs
        )

        self.ppg_criterion = torch.nn.MSELoss()

    def compute_loss(self,
                     x: Union[np.ndarray, torch.Tensor],
                     y: np.ndarray, **kwargs) -> np.ndarray:
        """
        Compute the gradient of the loss function w.r.t. `x`.

        :param x: Samples of shape (nb_samples, seq_length). Note that, it is allowable that sequences in the batch
                  could have different lengths. A possible example of `x` could be:
                  `x = np.array([np.array([0.1, 0.2, 0.1, 0.4]), np.array([0.3, 0.1])])`.
        :param y: Target values of shape (nb_samples). Each sample in `y` is a string and it may possess different
                  lengths. A possible example of `y` could be: `y = np.array(['SIXTY ONE', 'HELLO'])`.
        :return: Loss gradients of the same shape as `x`.
        """
        # x_in = torch.empty(len(x), device=x.device)
        x_in = x

        # Put the model in the training mode, otherwise CUDA can't backpropagate through the model.
        # However, model uses batch norm layers which need to be frozen
        self.DP_model.train()
        self.set_batchnorm(train=False)

        # Apply preprocessing
        x_preprocessed, y_preprocessed = self._apply_preprocessing(x_in, y, fit=False)

        # Transform data into the model input space
        inputs, targets, input_rates, target_sizes, _ = self._transform_model_input(
            x=x_preprocessed, y=y_preprocessed, compute_gradient=False
        )

        # Compute real input sizes
        input_sizes = input_rates.mul_(inputs.size()[-1]).int()

        # Call to DeepSpeech model for prediction
        outputs, output_sizes = self.DP_model(inputs.to(self._device), input_sizes.to(self._device))
        outputs = outputs.transpose(0, 1)

        if self._version == 2:
            outputs = outputs.float()
        else:
            outputs = outputs.log_softmax(-1)

        # Compute the loss
        loss = self.criterion(outputs, targets, output_sizes, target_sizes).to(self._device)
        # loss.backward()
        return loss

    def compute_ppg(self,
                     x: Union[np.ndarray, torch.Tensor], **kwargs) -> np.ndarray:
        """
        Compute the gradient of the loss function w.r.t. `x`.

        :param x: Samples of shape (nb_samples, seq_length). Note that, it is allowable that sequences in the batch
                  could have different lengths. A possible example of `x` could be:
                  `x = np.array([np.array([0.1, 0.2, 0.1, 0.4]), np.array([0.3, 0.1])])`.
        :param y: Target values of shape (nb_samples). Each sample in `y` is a string and it may possess different
                  lengths. A possible example of `y` could be: `y = np.array(['SIXTY ONE', 'HELLO'])`.
        :return: Loss gradients of the same shape as `x`.
        """
        # x_in = torch.empty(len(x), device=x.device)
        x_in = x

        # Put the model in the training mode, otherwise CUDA can't backpropagate through the model.
        # However, model uses batch norm layers which need to be frozen
        self.DP_model.train()
        self.set_batchnorm(train=False)

        # Apply preprocessing
        x_preprocessed, _ = self._apply_preprocessing(x_in, y=None, fit=False)

        # Transform data into the model input space
        inputs, _, input_rates, _, _ = self._transform_model_input(
            x=x_preprocessed, compute_gradient=False
        )

        # Compute real input sizes
        input_sizes = input_rates.mul_(inputs.size()[-1]).int()

        # Call to DeepSpeech model for prediction
        outputs, output_sizes = self.DP_model(inputs.to(self._device), input_sizes.to(self._device))
        # outputs = outputs.transpose(0, 1)

        # if self._version == 2:
        #     outputs = outputs.float()
        # else:
        #     outputs = outputs.log_softmax(-1)

        # Compute the loss
        # loss = self.criterion(outputs, targets, output_sizes, target_sizes).to(self._device)
        # loss.backward()
        return outputs

    def compute_ppgctc_loss(self,
                             x: Union[np.ndarray, torch.Tensor],
                             y: np.ndarray,
                             ppg, **kwargs) -> np.ndarray:
        """
        Compute the gradient of the loss function w.r.t. `x`.

        :param x: Samples of shape (nb_samples, seq_length). Note that, it is allowable that sequences in the batch
                  could have different lengths. A possible example of `x` could be:
                  `x = np.array([np.array([0.1, 0.2, 0.1, 0.4]), np.array([0.3, 0.1])])`.
        :param y: Target values of shape (nb_samples). Each sample in `y` is a string and it may possess different
                  lengths. A possible example of `y` could be: `y = np.array(['SIXTY ONE', 'HELLO'])`.
        :return: Loss gradients of the same shape as `x`.
        """
        # x_in = torch.empty(len(x), device=x.device)
        x_in = x

        # Put the model in the training mode, otherwise CUDA can't backpropagate through the model.
        # However, model uses batch norm layers which need to be frozen
        self.DP_model.train()
        self.set_batchnorm(train=False)

        # Apply preprocessing
        x_preprocessed, y_preprocessed = self._apply_preprocessing(x_in, y, fit=False)

        # Transform data into the model input space
        inputs, targets, input_rates, target_sizes, _ = self._transform_model_input(
            x=x_preprocessed, y=y_preprocessed, compute_gradient=False
        )

        # Compute real input sizes
        input_sizes = input_rates.mul_(inputs.size()[-1]).int()

        # Call to DeepSpeech model for prediction
        outputs, output_sizes = self.DP_model(inputs.to(self._device), input_sizes.to(self._device))
        ppg_loss = self.ppg_criterion(outputs, ppg)
        outputs = outputs.transpose(0, 1)

        if self._version == 2:
            outputs = outputs.float()
        else:
            outputs = outputs.log_softmax(-1)

        # Compute the loss
        ctc_loss = self.criterion(outputs, targets, output_sizes, target_sizes).to(self._device)
        # loss.backward()
        return ppg_loss, ctc_loss

    def compute_ppgloss(self,
                             x: Union[np.ndarray, torch.Tensor],
                             ppg, **kwargs) -> np.ndarray:
        """
        Compute the gradient of the loss function w.r.t. `x`.

        :param x: Samples of shape (nb_samples, seq_length). Note that, it is allowable that sequences in the batch
                  could have different lengths. A possible example of `x` could be:
                  `x = np.array([np.array([0.1, 0.2, 0.1, 0.4]), np.array([0.3, 0.1])])`.
        :param y: Target values of shape (nb_samples). Each sample in `y` is a string and it may possess different
                  lengths. A possible example of `y` could be: `y = np.array(['SIXTY ONE', 'HELLO'])`.
        :return: Loss gradients of the same shape as `x`.
        """
        # x_in = torch.empty(len(x), device=x.device)
        outputs = self.compute_ppg(x)
        ppg_loss = self.ppg_criterion(outputs, ppg)

        if torch.isnan(outputs).any():
            print('Warning: output ppg has NaN values')
        if torch.isnan(ppg).any():
            print('Warning: groundtruth ppg has NaN values')

        return ppg_loss
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    _date = '{}'.format(datetime.now().strftime("%m%d"))
    now = '{}'.format(datetime.now().strftime("%H%M"))

    # I/O parameters
    parser.add_argument('--input_wav', type=str, default = '',help='input wav. file')
    parser.add_argument('--input_pulse', type=str, default = '',help='input physical emitted. file')
    parser.add_argument('--output_wav', type=str, default=f"results/{_date}/{__file__.replace('.py','')}/output_{now}.wav", help='output adversarial wav. file')
    parser.add_argument('--model_path', type=str, default="<DeepSpeech 2 PyTorch's pretrained model>/librispeech_pretrained_v2.pth", required=True ,help='model pth path; better to use absolute path')
    parser.add_argument('--device', type=str, default='0', help='device')
    
    # attack parameters
    parser.add_argument('--target_sentence', type=str, default=np.asarray([" OPEN THE DOOR "]) , help='Please use uppercase')
    parser.add_argument('--attack_type', type=str, default="Universal_robust", help='[Universal_robust, Mute_robust]')
    parser.add_argument('--epsilon', type=float, default=1, help='epsilon pulse, the maximum is 1, meaning no constrain')
    parser.add_argument('--alpha', type=float, default=1e-1, help='alpha')
    parser.add_argument('--PGD_iter', type=int, default=800, help='PGD iteration times')
    parser.add_argument('--filter_freq', type=int, default=1000)
    parser.add_argument('--attack_length', type=float, default=1.2, help='attack length: e.g., 0.5s, 1.1s')
    parser.add_argument('--sync_range', type=float, default=0.08, help='(s), you can set the random value for training from 0-100 ms')
    parser.add_argument('--amp_range', type=float, default=1, help='[1,2,3,4,5]')
    parser.add_argument('--ir_amp_range', type=float, default=1, help='[1,2,3,4,5], the varying range of UFRs (1 means 0~1)')
    parser.add_argument('--ir_base', type=float, default=0.5, help='[1,2,3,4,5], the minimum UFR amplitude')

    parser.add_argument('--time_aug', type=int, default=1, help='Time Robust Training')
    parser.add_argument('--amp_aug', type=int, default=1, help='Amplitude Robust Training')
    parser.add_argument('--ir_aug', type=int, default=6, help='Selecting 6 pairs of UFRs+anomalous noise from the pool')
    parser.add_argument('--shuffle', type=bool, default=False, help='')
    
    parser.add_argument('--ir_path', type=str, default="data/dev_channel_ufr/IR_trunc", help='the path of UFRs')
    parser.add_argument('--noise_path', type=str, default="data/dev_channel_noise", help='the path of ultrasound-channel noise')
    parser.add_argument('--mute_path', type=str, default="", help='the already-trained silence pertubation')
    
    parser.add_argument('--receiver', type=str, default="", help='the target device')

    parser.add_argument('--ir_opt', type=bool, default=True, help='True: use a uniform distribution to mimic the energy variance in location-variable attacks')

    # plot parameters
    parser.add_argument('--plot_ori_spec', type=str, default="None", help='Path to save the original spectrogram')
    parser.add_argument('--plot_adv_spec', type=str, default="None", help='Path to save the adversarial spectrogram')

    args = parser.parse_args()
    if 'mute' in args.attack_type:
        args.target_sentence = np.asarray([" "]); args.attack_length = 5
    else:
        args.target_sentence = np.asarray([" OPEN THE DOOR "]); args.attack_length = 1.2

    print(f'The targeted sentence is: {args.target_sentence}')

    os.environ["CUDA_VISIBLE_DEVICES"]=args.device
    gpu_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    import torch, torchaudio

    cfg = TranscribeConfig
    model = DeepSpeech4Loss(pretrained_model="librispeech", device="cuda:0") # greedy search, quick
    # model = DeepSpeech4Loss(pretrained_model="librispeech", device="cuda:0", lm_path="../../adversarial_asr/deepspeech_pytorch/LanguageModel/3-gram.pruned.3e-7.arpa", decoder_type="beam", alpha=1.97, beta=4.36, beam_width=1024) # beam search, slow but accurate
    
    receiver_type = args.noise_path.split('/')[-1].split('_')[0] if args.receiver == "" else args.receiver
    run_name = f"{args.target_sentence[0]}_{receiver_type}_len-{args.attack_length}_amp-{args.amp_range}_iramp-{args.ir_amp_range}_sync-{args.sync_range}_{args.PGD_iter}_{_date}_{now}/"
    args.output_wav = f"results/{_date}/{__file__.replace('.py','')}/{run_name}"
    os.makedirs(args.output_wav, exist_ok=True)

    wandb.init(name=run_name)
    wandb.config.update(args)

    for num in range(1):
        attacker = Attacker(model=model, sound=None, target=args.target_sentence, device=gpu_device, save=args.output_wav)
        attacker.attack(args)
    if args.plot_ori_spec != "None":
        attacker.get_ori_spec(args.plot_ori_spec)
    
    if args.plot_adv_spec != "None":
        attacker.get_adv_spec(args.plot_adv_spec)

