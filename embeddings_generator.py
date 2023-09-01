import torch,random
import librosa
import yaml
from retrieval.models.ase_model import ASE
import pickle
import json
import os
from tqdm import tqdm
import argparse, math
import json
import pandas as pd
from re import sub


def text_preprocess(sentence):

    # transform to lower case
    sentence = sentence.lower()

    # remove any forgotten space before punctuation and double space
    sentence = sub(r'\s([,.!?;:"](?:\s|$))', r'\1', sentence).replace('  ', ' ')

    # remove punctuations
    # sentence = sub('[,.!?;:\"]', ' ', sentence).replace('  ', ' ')
    sentence = sub('[(,.!?;:|*\")]', ' ', sentence).replace('  ', ' ')
    return sentence

def Extract_embeddings(model,data_path,text_or_not,config):

    h5file_df = pd.read_csv(os.path.join(data_path,"wav.csv"), sep="\t")
    h5file_dict = dict(zip(h5file_df["audio_id"], h5file_df["file_name"]))
    caption_info = json.load(open(os.path.join(data_path,"text.json"), "r"))["audios"]

    model.eval()
    out_data = []
    with torch.no_grad(), tqdm(total=len(caption_info), ncols=100,
                                ascii=True) as pbar:
        for audio_caps in caption_info:

            if not os.path.exists(h5file_dict[audio_caps["audio_id"]]):
                print(audio_caps["audio_id"])
            audio_data, _ = librosa.load(h5file_dict[audio_caps["audio_id"]], sr=config["audio_args"]["sr"], mono=True) # sample rate should be 48000
            if audio_data.shape[0]<=0:
                continue
            if config["audio_args"]["max_length"]!=0:
                # print(config["audio_args"]["max_length"]*config["audio_args"]["sr"])
                # print(audio_data.shape[-1])
                if audio_data.shape[-1] > config["audio_args"]["max_length"]*config["audio_args"]["sr"]:
                    max_start = audio_data.shape[-1] - config["audio_args"]["max_length"]*config["audio_args"]["sr"]
                    start = random.randint(0, max_start)
                    audio_data = audio_data[start: start + config["audio_args"]["max_length"]*config["audio_args"]["sr"]]
            audio_data = audio_data.reshape(1, -1)
            # print(audio_data.shape)
            audio_embed = model.encode_audio(torch.tensor(audio_data).to('cuda:2'))

            if text_or_not == True:
                for caption in audio_caps["captions"]:
                    text_embed = model.encode_text(text_preprocess(caption["caption"]))
                    out_data.append({"audio_embedding":audio_embed, "caption":caption["caption"], "text_embedding":text_embed,"audio_id":audio_caps["audio_id"]})
            else:
                # print(audio_caps["captions"])
                out_data.append({"audio_embedding":audio_embed, "caption":audio_caps["captions"], "text_embedding":0,"audio_id":audio_caps["audio_id"]})
            pbar.update()
    
    return out_data

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="retrieval/settings/extract.yaml", type=str,help="Setting files")
    parser.add_argument("-t", "--model_type", default="cnn", type=str,
                    help="Model type.")
    parser.add_argument("-m", "--model", default="Cnn14", type=str,help="Model name.")
    parser.add_argument("-a", "--max_length", default=10, type=int,help="Max length.")
    parser.add_argument('--clap_model_ckpt', default="/home/zhangyiming/clap/HTSAT-BERT-ZS.pt")
    parser.add_argument('--dataset_mode', type=float, default=0)  # 0 for AudioCaps, 1 for Clotho.
    parser.add_argument('--dataset_path',  default="/home/zhangyiming/Data-processed")  
    parser.add_argument('--out_path',  default="/home/zhangyiming/clap/clap_embedding_wavcaps")  
    parser.add_argument('--device',  default="cuda:2")  
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)
    config["audio_encoder_args"]["type"] = args.model_type
    config["audio_encoder_args"]["model"] = args.model
    config["audio_args"]["max_length"] = args.max_length

    # Load the Model
    model = ASE(config)
    model = model.to(args.device)
    model.eval()
    state_dict = torch.load(args.clap_model_ckpt, map_location=args.device)
    # print(state_dict["model"].keys())
    model.load_state_dict(state_dict["model"])

    if args.dataset_mode == 0:
        base_data_path = os.path.join(args.dataset_path,"audiocaps")
        out_path = os.path.join(args.out_path,"audiocaps")
    elif args.dataset_mode == 1:
        base_data_path = os.path.join(args.dataset_path,"clotho")
        out_path = os.path.join(args.out_path,"clotho")
    else:
        print("Warining!!! No DataSet!!!")
    
    for split in ["development","evaluation","validation"]:
        data_path = os.path.join(base_data_path,split)
        print(f"---Extract the embeddings of {split} set---")
        out_data = Extract_embeddings(model,data_path,split=="development",config)
        os.makedirs(os.path.join(out_path,split),exist_ok=True)
        with open(os.path.join(out_path,split,"embedding.pkl"), 'wb') as f:
            pickle.dump(out_data,f)


if __name__ == '__main__':
    main()