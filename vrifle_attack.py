import torch
import torchaudio
import numpy as np
import random
import torch.optim as optim
from warpctc_pytorch import CTCLoss
import glob
import pandas as pd
import wandb
import os

def early_stop(monitor,value,patience,times):
    if(monitor < value):
        times[0] += 1
    if times[0] >= patience:
        return True
    return False


class Attacker:
    def __init__(self, model, sound, target, sample_rate=16000, device="cpu", save=None):
        """
        model: deepspeech model
        sound: raw sound data [-1 to +1] (read from torchaudio.load)
        label: string
        """
        self.sound = sound
        self.sample_rate = sample_rate
        self.target = target
        self.model = model
        self.criterion = CTCLoss()
        self.device = device
        self.save = save

    def attack(self, args):
        print("Start attack")
        # We showcase the core code for the digital domain attacks, where the unlimited amplitude advantages of VRifle enables it to mute and alter victim's speech easily.
        # By integrating the RIR noise database, and convolving with victim's speech as most codes have done, it should well mimic the real-world attack scenarios.
        if args.attack_type == "Mute_robust":
            log_table = wandb.Table(columns=['epoch', 'train_loss', 'val_loss', 'predict_result'])
            times = [0]
            ######################### Load the Necessray Components in Ultrasonic Transformation Modeling #############################
            ######################### (1) Ultrasonic Anomalous Noise #############################
            ######################### (2) Ultrasonic Frequency Response #############################
            # Robust Training：Employing multiple (1)+(2) pairs for training
            noises=[]
            sig_irs=[]; sig_irs_train=[]; sig_irs_val=[]
            assert 'trunc' in args.ir_path.lower()
            ir_list = glob.glob(f"{args.ir_path}/*.wav")
            for _, ir in enumerate(ir_list):
                ir, sr = torchaudio.load(ir)
                sig_irs.append(torch.fliplr(ir).to(self.device))
            noise_list = glob.glob(f"{args.noise_path}/*.wav")
            for _, ns in enumerate(noise_list):
                noise, sr = torchaudio.load(ns)
                noises.append(noise.to(self.device))

            if args.shuffle:
                random.shuffle(sig_irs)
            sig_irs_train = sig_irs[:int(len(sig_irs)*0.9)] # We use 90% UFRs for training
            sig_irs_val = sig_irs[int(len(sig_irs)*0.9):] # 10% for testing
            # Set the length for attack perturbation duration
            attack_length = int(args.attack_length * self.sample_rate)
            commands_list = []
            # Randomly select 10 speakers from the Fluent Speech Command Dataset, each speaker contributes 100 samples for training
            val_speakers = ["7NqqnAOPVVSKnxyv","8B9N9jOOXGUordVG","9MX3AgZzVgCw4W4j","D4jGxZ7KamfVo4E2V","DWNjK4kYDACjeEg3","eBQAWmMg4gsLYLLa","mj4BWeRbp7ildyB9d","NgXwdx5KkZI5GRWa","Pz327QrLaGuxW8Do","vnljypgejkINbBAY"]
            for speakers in val_speakers:
                commands_list0 = sorted(glob.glob('../../data/fluent_speech_commands_dataset/wavs_vad/speakers/' + speakers + '/*.wav'))[:100]
                commands_list = commands_list + commands_list0
            commands_list = random.sample(commands_list, 100)
            print(len(commands_list))

            if args.input_pulse != '':
                uni_pulse = torchaudio.load(args.input_pulse)[0]
            else:    
                uni_pulse = (args.epsilon / 0.5) * (torch.rand(1, attack_length) - 0.5)
            uni_pulse = uni_pulse.to(self.device)
            uni_pulse.requires_grad = True
            opt = optim.Adam([uni_pulse], lr=args.alpha)
            
            for epoch in range(args.PGD_iter):
                print(f"pulse processing ...  {epoch+1} / {args.PGD_iter}", end="\r")
                loss_log = 0
                
                # The audio to be optimized
                opt.zero_grad()
                for h_batch in range(10):
                    loss_arr = []
                    for h_iter in range(10): # To avoid the GPU out-of-memory error, we divide the batch into small batches.
                        idx = h_batch*10 + h_iter
                        data_raw = torchaudio.load(commands_list[idx])[0]
                        if (data_raw.shape[1]>(args.attack_length-args.sync_range)*self.sample_rate):
                            data_raw = torchaudio.load(commands_list[idx])[0][:,:int((args.attack_length-args.sync_range)*self.sample_rate)]
                        ''' Diagram of the Mute Mode 
                            ################ victim ############|#### go on speaking ###
                              ############## mute #######################################################
                              ############## noise ######################################################
                        '''
                        for j in range(args.time_aug):
                            data_raw = data_raw.to(self.device)
                            start = int(random.uniform(0, args.sync_range)*self.sample_rate)
                            end = int(start+data_raw.shape[1])

                            # Amplitude Robustness
                            for k in range(args.amp_aug):
                                noise = noises[random.randint(0,len(noises)-1)]
                                noise_start = random.randint(0,noise.shape[1]-int((args.attack_length+args.sync_range)*self.sample_rate))
                                noise_end = noise_start + attack_length
                                noise_random = noise[:, noise_start:noise_end]

                                vic_amp = (args.amp_range * random.random()) + 0.5 # Mimicking the distance variance of the speaker and the device
                                for ir_idx in np.random.choice(len(sig_irs_train), size=args.ir_aug):
                                    if args.ir_opt:
                                        ir_amp = random.random()*args.ir_amp_range + args.ir_base # mimic the UFR amplitude varies with distance                                   
                                    pulse_adv_low = ir_amp * torch.nn.functional.conv1d(uni_pulse.view(1,1,-1), sig_irs_train[ir_idx].view(1,1,-1), padding='same').view(1,-1)
                                    data = torch.concat((pulse_adv_low[:,:start], data_raw+pulse_adv_low[:,start:end], pulse_adv_low[:,end:]),dim=1)+noise_random
                                    # You can also try to modify the code into
                                    ''' Diagram of the Mute Mode 
                                            ################ victim ############|#### go on speaking ###
                                        ############## mute #######################################################
                                        ############## noise ######################################################
                                    '''
                                    data.data = torch.clamp(data, min=-1, max=1)

                                    loss_arr.append(self.model.compute_loss(data, self.target))
                            # torch.cuda.empty_cache()
                    for loss in loss_arr:
                        loss = loss / (args.time_aug * args.amp_aug * args.ir_aug)
                        loss_log += loss.detach().cpu().numpy().item()
                        loss.backward()
                opt.step()

                uni_pulse.data = torch.clamp(uni_pulse, min=-args.epsilon, max=args.epsilon)

                wandb.log({"train_loss":loss_log}, step=epoch)

                # ----------------------- start validation ----------------------------
                if (epoch+1) % 10 == 0:
                    loss_val=[]; predict_val = ""
                    loss_val_log=0

                    for ir_idx in np.arange(len(sig_irs_val)):
                        pulse_adv_low = torch.nn.functional.conv1d(uni_pulse.view(1,1,-1), sig_irs_val[ir_idx].view(1,1,-1), padding='same').view(1,-1)
                        data = torch.concat((pulse_adv_low[:,:start], data_raw+pulse_adv_low[:,start:end], pulse_adv_low[:,end:]),dim=1)+noise_random
                        data.data = torch.clamp(data, min=-1, max=1)
                        loss_val.append(self.model.compute_loss(data, self.target))
                        decoded_output = self.model.predict(data.detach().cpu().numpy(),transcription_output=True)
                        predict_val = " | ".join((predict_val, decoded_output[0]))
                    for loss in loss_val:
                        loss = loss / (len(sig_irs_val))
                        loss_val_log += loss.detach().cpu().numpy().item()

                    wandb.log({"val_loss":loss_val_log}, step=epoch)
                    
                    print(f"epoch: {epoch},cur_epoch_loss: {loss_log}, Adversarial prediction: {decoded_output},val_loss:{loss_val_log}")
                    log_table.add_data(epoch, loss_log, loss_val_log, predict_val)
                    if self.save:
                        self_transcript = self.model.predict(pulse_adv_low.detach().cpu().numpy(),transcription_output=True)[0]
                        torchaudio.save(os.path.join(self.save, f'mute_{epoch}_{self_transcript}.wav'), src=uni_pulse.cpu(), sample_rate=self.sample_rate)
                        torchaudio.save(os.path.join(self.save, f'mute_low_{epoch}_{self_transcript}.wav'), src=pulse_adv_low.cpu(), sample_rate=self.sample_rate)
                
                if early_stop(monitor = loss_log,value = 0.3, patience=5,times=times):
                    print("loss is small enough so stop train early!")
                    break
            
            wandb.log({"table": log_table})


        if args.attack_type == "Universal_robust":
            log_table = wandb.Table(columns=['epoch', 'train_loss', 'val_loss', 'predict_result'])
            times = [0]
            ######################### Load the Necessray Components in Ultrasonic Transformation Modeling #############################
            ######################### (1) Ultrasonic Anomalous Noise #############################
            ######################### (2) Ultrasonic Frequency Response #############################
            # Robust Training：Employing multiple (1)+(2) pairs for training
            noises=[]
            sig_irs=[]; sig_irs_train=[]; sig_irs_val=[]
            ir_list = glob.glob(f"{args.ir_path}/*.wav")
            for _, ir in enumerate(ir_list):
                ir, sr = torchaudio.load(ir)
                sig_irs.append(torch.fliplr(ir).to(self.device))
            noise_list = glob.glob(f"{args.noise_path}/*.wav")
            for _, ns in enumerate(noise_list):
                noise, sr = torchaudio.load(ns)
                noises.append(noise.to(self.device))

            if args.shuffle:
                random.shuffle(sig_irs)
            sig_irs_train = sig_irs[:int(len(sig_irs)*0.9)]
            sig_irs_val = sig_irs[int(len(sig_irs)*0.9):]
            silence_pulse = torchaudio.load(args.mute_path)[0]; silence_pulse = silence_pulse.to(self.device) # load the well-trained silence perturbation
            attack_length = int(args.attack_length * self.sample_rate)
            link_length = attack_length + int(2.5 * self.sample_rate)
            commands_list = []
            val_speakers = ["7NqqnAOPVVSKnxyv","8B9N9jOOXGUordVG","9MX3AgZzVgCw4W4j","D4jGxZ7KamfVo4E2V","DWNjK4kYDACjeEg3","eBQAWmMg4gsLYLLa","mj4BWeRbp7ildyB9d","NgXwdx5KkZI5GRWa","Pz327QrLaGuxW8Do","vnljypgejkINbBAY"]
            for speakers in val_speakers:
                commands_list0 = sorted(glob.glob('../../data/fluent_speech_commands_dataset/wavs_vad/speakers/' + speakers + '/*.wav'))[:100] # load the well-trained silence perturbation
                commands_list = commands_list + commands_list0
            commands_list = random.sample(commands_list, 200)
            if args.input_pulse != '':
                uni_pulse = torchaudio.load(args.input_pulse)[0]
            else:
                uni_pulse = torch.zeros(1, attack_length)
            uni_pulse = uni_pulse.to(self.device)
            uni_pulse.requires_grad = True
            opt = optim.Adam([uni_pulse], lr=args.alpha)
            for epoch in range(args.PGD_iter):
                print(f"pulse processing ...  {epoch+1} / {args.PGD_iter}", end="\r")
                loss_log = 0
                opt.zero_grad()
                for h_batch in range(20):
                    loss_arr = []
                    for h_iter in range(10):
                        idx = h_batch*10+h_iter
                        data_raw = torchaudio.load(commands_list[idx])[0]
                        if data_raw.shape[1]>link_length-int(args.sync_range*self.sample_rate):
                            data_raw = data_raw[:,:link_length-int(args.sync_range*self.sample_rate)]
                        data_raw = data_raw.to(self.device)
                        for j in range(args.time_aug):
                            start = int(random.uniform(0, args.sync_range)*self.sample_rate)
                            end = int(data_raw.shape[1])

                            for k in range(args.amp_aug):
                                noise = noises[random.randint(0,len(noises)-1)]
                                noise_start = random.randint(0,noise.shape[1]-int(args.sync_range*self.sample_rate)-link_length)
                                noise_end  = noise_start + link_length
                                noise_random = noise[:, noise_start:noise_end]
                                vic_amp = (args.amp_range * random.random()) + 0.5
                                for ir_idx in np.random.choice(len(sig_irs_train), size=args.ir_aug):
                                    if args.ir_opt:
                                        ir_amp = random.random()*args.ir_amp_range + args.ir_base
                                    
                                    pulse_adv_low = ir_amp * torch.nn.functional.conv1d(torch.concat((uni_pulse,silence_pulse),dim=1).view(1,1,-1), sig_irs_train[ir_idx].view(1,1,-1), padding='same').view(1,-1)
                                    
                                    data = torch.concat((vic_amp*data_raw[:,:start], vic_amp*data_raw[:,start:end]+pulse_adv_low[:,:end-start]+noise_random[:,:end-start], pulse_adv_low[:,end-start:]+noise_random[:,end-start:]),dim=1)
                                    data.data = torch.clamp(data, min=-1, max=1)

                                    loss_arr.append(self.model.compute_loss(data, self.target))

                    for loss in loss_arr:
                        loss = loss / (args.time_aug * args.amp_aug * args.ir_aug * 3)
                        loss_log += loss.detach().cpu().numpy().item()
                        loss.backward(retain_graph=True)
                opt.step()

                uni_pulse.data = torch.clamp(uni_pulse, min=-args.epsilon, max=args.epsilon)

                wandb.log({"train_loss":loss_log}, step=epoch)

                # ----------------------- start validation ----------------------------
                if (epoch+1) % 10 == 0:
                    if self.save:
                        self_transcript = self.model.predict(pulse_adv_low.detach().cpu().numpy(),transcription_output=True)[0]
                        torchaudio.save(os.path.join(self.save, f'link_{epoch}_{self_transcript}.wav'), src=torch.concat((uni_pulse,silence_pulse),dim=1).cpu(), sample_rate=self.sample_rate)
                        torchaudio.save(os.path.join(self.save, f'out_{epoch}_{self_transcript}.wav'), src=uni_pulse.cpu(), sample_rate=self.sample_rate)
                        torchaudio.save(os.path.join(self.save, f'link_low_{epoch}_{self_transcript}.wav'), src=pulse_adv_low.cpu(), sample_rate=self.sample_rate)
                    
                    loss_val=[]; predict_val = ""
                    loss_val_log=0
                    for ir_idx in np.arange(len(sig_irs_val)):
                        pulse_adv_low = torch.nn.functional.conv1d(torch.concat((uni_pulse,silence_pulse),dim=1).view(1,1,-1), sig_irs_val[ir_idx].view(1,1,-1), padding='same').view(1,-1)
                        data = torch.concat((data_raw[:,:start], data_raw[:,start:end]+pulse_adv_low[:,:end-start]+noise_random[:,:end-start], pulse_adv_low[:,end-start:]+noise_random[:,end-start:]),dim=1)
                        data.data = torch.clamp(data, min=-1, max=1)

                        loss_val.append(self.model.compute_loss(data, self.target))
                        decoded_output = self.model.predict(data.detach().cpu().numpy(),transcription_output=True)
                        predict_val = " | ".join((predict_val, decoded_output[0]))
                    for loss in loss_val:
                        loss = loss / (len(sig_irs_val))
                        loss_val_log += loss.detach().cpu().numpy().item()

                    wandb.log({"val_loss":loss_val_log}, step=epoch)
                    
                    print(f"epoch: {epoch},cur_epoch_loss: {loss_log}, Adversarial prediction: {decoded_output},val_loss:{loss_val_log}")
                    log_table.add_data(epoch, loss_log, loss_val_log, predict_val)
                
                if early_stop(monitor = loss_log,value = 0.3, patience=5,times=times):
                    print("loss is small enough so stop train early!")
                    break
            wandb.log({"table": log_table})