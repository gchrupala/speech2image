import sys
sys.path.append('./PyTorch/functions')
sys.path.append('./preprocessing')
from encoders import img_encoder, audio_rnn_encoder
import torch
import logging
import glob
import numpy as np

audio_config = {'conv':{'in_channels': 39, 'out_channels': 64, 'kernel_size': 6, 'stride': 2,
               'padding': 0, 'bias': False}, 'rnn':{'input_size': 64, 'hidden_size': 1024, 
               'num_layers': 4, 'batch_first': True, 'bidirectional': True, 'dropout': 0}, 
               'att':{'in_size': 2048, 'hidden_size': 128, 'heads': 1, 'scalar': False}}
# automatically adapt the image encoder output size to the size of the caption encoder
out_size = audio_config['rnn']['hidden_size'] * 2**audio_config['rnn']['bidirectional'] * audio_config['att']['heads']
image_config = {'linear':{'in_size': 2048, 'out_size': out_size}, 'norm': True}



def embed_img(img_net, images):
    model = img_encoder(image_config)
    model.load_state_dict(torch.load(img_net))
    for p in model.parameters():
    	p.requires_grad = False
    model.eval()
    data = preprocess_img(images)
    return model(data)


def embed_audio(audio_net, audios):    
    model = audio_rnn_encoder(audio_config)
    model.load_state_dict(torch.load(audio_net))
    for p in model.parameters():
    	p.requires_grad = False
    model.eval()
    data = preprocess_audio(audios)
    output = []
    for cap in data:
        cap = torch.FloatTensor(cap).t().unsqueeze(0)
        emb = model(cap, [cap.size(2)])
        output.append(emb)
    return torch.cat(output)


def preprocess_img(images):
    import torch.nn as nn
    import torchvision.models as models
    import torchvision.transforms as transforms
    import PIL.Image
    model = models.resnet152(pretrained = True)
    model = nn.Sequential(*list(model.children())[:-1])
    model = model.cuda()
    for p in model.parameters():
    	p.requires_grad = False
    model.eval()
    tencrop = transforms.TenCrop(224)
    tens = transforms.ToTensor()
    normalise = transforms.Normalize(mean = [0.485,0.456,0.406], 
                                     std = [0.229, 0.224, 0.225])
    resize = transforms.Resize(256, PIL.Image.ANTIALIAS)
    vis_array = []
    for image in images:
        logging.info("Processing {}".format(image))
        im = tencrop(resize(PIL.Image.open(image)))
        im = torch.cat([normalise(tens(x)).unsqueeze(0) for x in im])
        im = torch.autograd.Variable(im).cuda()
        if not im.size()[1] == 3:
            im = im.expand(im.size()[0], 3, im.size()[2], im.size()[3])
        activations = model(im).mean(0).squeeze()
        vis_array.append(activations.unsqueeze(0))
    return torch.cat(vis_array).cpu()

def preprocess_audio(audios):
    from aud_feat_functions import get_fbanks, get_freqspectrum, get_mfcc, delta, raw_frames
    from scipy.io.wavfile import read
    import numpy
    feat = 'mfcc'
    params = []
    # set alpha for the preemphasis
    alpha = 0.97
    # set the number of desired filterbanks
    nfilters = 40
    # windowsize and shift in seconds
    t_window = .025
    t_shift = .010
    # option to include delta and double delta features
    use_deltas = True
    # option to include frame energy
    use_energy = True
    # put paramaters in a list
    params.append(alpha)
    params.append(nfilters) 
    params.append(t_window)
    params.append(t_shift)
    params.append(feat)
    params.append(None)
    params.append(use_deltas)
    params.append(use_energy)
    output = []
    for cap in audios:
            logging.info("Processing {}".format(cap))
            # read audio samples
            try:
                input_data = read(cap)


        
                # sampling frequency
                fs = input_data[0]
                # get window and frameshift size in samples
                window_size = int(fs*params[2])
                frame_shift = int(fs*params[3])
        
                # create features (implemented are raw audio, the frequency spectrum, fbanks and
                # mfcc's)
            
                
                [frames, energy] = raw_frames(input_data, frame_shift, window_size)
                freq_spectrum = get_freqspectrum(frames, params[0], fs, window_size)
                fbanks = get_fbanks(freq_spectrum, params[1], fs)
                features = get_mfcc(fbanks)
            
                # optionally add the frame energy
                if params[7]:
                    features = numpy.concatenate([energy[:,None], features],1)
                    # optionally add the deltas and double deltas
                if params[6]:
                    single_delta= delta (features,2)
                    double_delta= delta(single_delta,2)
                    features= numpy.concatenate([features,single_delta,double_delta],1)
            except ValueError:
                logging.warn("Could not read file {}".format(cap))
                features = numpy.zeros_like(output[-1]) #FIXME outputting random crap here
            # append new data to the tables
            output.append(features)
    return output
        

def validation():
    """Returns the validation set information:
    - array of audio paths
    - array of image paths
    - array correct[i][j] indicating whether image j is correct for caption i.
    """
    import json
    mapping = {}
    for line in open("/roaming/gchrupal/datasets/flickr8k/wav2capt.txt"):
        wav, jpg, _ = line.split()
        mapping[jpg] = mapping.get(jpg, []) + [wav]
    audio_path = []
    image_path = []
    correct = np.zeros((1000,5000), dtype=bool)
    dataset = json.load(open('/roaming/gchrupal/datasets/flickr8k/dataset.json'))
    for image in dataset['images']:
        if image['split'] == 'val':
            image_path.append(image['filename'])
            for wav in mapping[image['filename']]:
                audio_path.append(wav)
                correct[len(image_path)-1][len(audio_path)-1] = True
    return np.array(audio_path), np.array(image_path), correct

def main():
    import pickle
    logging.basicConfig(level=logging.INFO)
    imagedir = '/roaming/gchrupal/datasets/flickr8k/Flickr8k_Dataset/Flicker8k_Dataset/'
    audiodir = '/roaming/gchrupal/datasets/flickr8k/flickr_audio/wavs/'

    au, im, co = validation() 
    
    
    emb_audio = embed_audio("experiments/a/caption_model.32", [audiodir + x for x in au])
    emb_image = embed_img("experiments/a/image_model.32",     [imagedir + x for x in im])
    result = dict(audiopath=au, imagepath=im, audio=emb_audio.numpy(), image=emb_image.numpy(), correct=co)
    pickle.dump(result, open("validation.pkl", 'wb'))
    
    

