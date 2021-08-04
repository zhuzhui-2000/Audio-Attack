import hydra
import torch
import numpy

from deepspeech_pytorch.configs.inference_config import EvalConfig
from deepspeech_pytorch.decoder import GreedyDecoder
from deepspeech_pytorch.loader.data_loader import SpectrogramDataset, AudioDataLoader
from deepspeech_pytorch.utils import load_model, load_decoder
from deepspeech_pytorch.validation import run_evaluation, train_evaluation, attack_test_evaluation







def evaluate(cfg: EvalConfig):
    device = torch.device("cuda" if cfg.model.cuda else "cpu")

    model = load_model(
        device=device,
        model_path=cfg.model.model_path
    )

    decoder = load_decoder(
        labels=model.labels,
        cfg=cfg.lm
    )
    target_decoder = GreedyDecoder(
        labels=model.labels,
        blank_index=model.labels.index('_')
    )
    test_dataset = SpectrogramDataset(
        plot = cfg.plot,
        attack = cfg.attack,
        input_word=cfg.input_word,
        target_word=cfg.target_word,
        audio_conf=model.spect_cfg,
        input_path=hydra.utils.to_absolute_path(cfg.test_path),
        labels=model.labels,
        normalize=True
    )
    test_loader = AudioDataLoader(
        test_dataset,
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers
    )
    #tensor_input_word = test_dataset.parse_target_word(cfg.input_word)
    #tensor_target_word = test_dataset.parse_target_word(cfg.target_word)
    #print(tensor_input_word)
    delta = numpy.load("/home/mmc-2018012484/deepspeech.pytorch/outputs/2021-07-27/00-29-19/delta.npy")
    delta = delta.reshape(1,1,delta.shape[0],delta.shape[1])
    delta = torch.tensor(delta).to(device)
    

    if (cfg.attack):
        if(cfg.plot):
            wer, cer = attack_test_evaluation(input_word=cfg.input_word,target_word=cfg.target_word,
                delta = delta,
                test_loader=test_loader,
                device=device,
                model=model,
                decoder=decoder,
                target_decoder=target_decoder,
                precision=cfg.model.precision
            )   
        else:
            wer, cer = train_evaluation(input_word=cfg.input_word,target_word=cfg.target_word, attack_length = cfg.attack_length,
                test_loader=test_loader,
                device=device,
                model=model,
                decoder=decoder,
                target_decoder=target_decoder,
                precision=cfg.model.precision
            )    
    else:   
        wer, cer = run_evaluation(
            test_loader=test_loader,
            device=device,
            model=model,
            decoder=decoder,
            target_decoder=target_decoder,
            precision=cfg.model.precision
        )

    print('Test Summary \t'
          'Average WER {wer:.3f}\t'
          'Average CER {cer:.3f}\t'.format(wer=wer, cer=cer))
