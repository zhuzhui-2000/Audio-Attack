import hydra
import torch

from deepspeech_pytorch.configs.inference_config import EvalConfig
from deepspeech_pytorch.decoder import GreedyDecoder
from deepspeech_pytorch.loader.data_loader import SpectrogramDataset, AudioDataLoader
from deepspeech_pytorch.utils import load_model, load_decoder
from deepspeech_pytorch.validation import run_evaluation, train_evaluation







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
    tensor_input_word = test_dataset.parse_target_word(cfg.input_word)
    tensor_target_word = test_dataset.parse_target_word(cfg.target_word)
    print(tensor_input_word)
    if (cfg.attack):
        wer, cer = train_evaluation(input_word=tensor_input_word,target_word=tensor_input_word,
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
