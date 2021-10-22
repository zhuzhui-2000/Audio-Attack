from abc import ABC, abstractmethod

import torch
from torch.cuda.amp import autocast
from torch.nn import CTCLoss, Module, _reduction, functional
from tqdm import tqdm
import torch.optim as optim
import numpy

from deepspeech_pytorch.decoder import Decoder, GreedyDecoder
from wavenet import WaveNet

from pytorch_lightning.metrics import Metric
import Levenshtein as Lev

import matplotlib.pyplot as plt

class My_Loss(Module):
    reduction: str

    def __init__(self, size_average=None, reduce=None, reduction: str = 'mean') -> None:
        super(My_Loss, self).__init__()
        if size_average is not None or reduce is not None:
            self.reduction: str = _reduction.legacy_get_string(size_average, reduce)
        else:
            self.reduction = reduction

class My_CTCLoss(My_Loss):
    r"""The Connectionist Temporal Classification loss.

    Calculates loss between a continuous (unsegmented) time series and a target sequence. CTCLoss sums over the
    probability of possible alignments of input to target, producing a loss value which is differentiable
    with respect to each input node. The alignment of input to target is assumed to be "many-to-one", which
    limits the length of the target sequence such that it must be :math:`\leq` the input length.

    Args:
        blank (int, optional): blank label. Default :math:`0`.
        reduction (string, optional): Specifies the reduction to apply to the output:
            ``'none'`` | ``'mean'`` | ``'sum'``. ``'none'``: no reduction will be applied,
            ``'mean'``: the output losses will be divided by the target lengths and
            then the mean over the batch is taken. Default: ``'mean'``
        zero_infinity (bool, optional):
            Whether to zero infinite losses and the associated gradients.
            Default: ``False``
            Infinite losses mainly occur when the inputs are too short
            to be aligned to the targets.

    Shape:
        - Log_probs: Tensor of size :math:`(T, N, C)`,
          where :math:`T = \text{input length}`,
          :math:`N = \text{batch size}`, and
          :math:`C = \text{number of classes (including blank)}`.
          The logarithmized probabilities of the outputs (e.g. obtained with
          :func:`torch.nn.functional.log_softmax`).
        - Targets: Tensor of size :math:`(N, S)` or
          :math:`(\operatorname{sum}(\text{target\_lengths}))`,
          where :math:`N = \text{batch size}` and
          :math:`S = \text{max target length, if shape is } (N, S)`.
          It represent the target sequences. Each element in the target
          sequence is a class index. And the target index cannot be blank (default=0).
          In the :math:`(N, S)` form, targets are padded to the
          length of the longest sequence, and stacked.
          In the :math:`(\operatorname{sum}(\text{target\_lengths}))` form,
          the targets are assumed to be un-padded and
          concatenated within 1 dimension.
        - Input_lengths: Tuple or tensor of size :math:`(N)`,
          where :math:`N = \text{batch size}`. It represent the lengths of the
          inputs (must each be :math:`\leq T`). And the lengths are specified
          for each sequence to achieve masking under the assumption that sequences
          are padded to equal lengths.
        - Target_lengths: Tuple or tensor of size :math:`(N)`,
          where :math:`N = \text{batch size}`. It represent lengths of the targets.
          Lengths are specified for each sequence to achieve masking under the
          assumption that sequences are padded to equal lengths. If target shape is
          :math:`(N,S)`, target_lengths are effectively the stop index
          :math:`s_n` for each target sequence, such that ``target_n = targets[n,0:s_n]`` for
          each target in a batch. Lengths must each be :math:`\leq S`
          If the targets are given as a 1d tensor that is the concatenation of individual
          targets, the target_lengths must add up to the total length of the tensor.
        - Output: scalar. If :attr:`reduction` is ``'none'``, then
          :math:`(N)`, where :math:`N = \text{batch size}`.

    Examples::

        >>> # Target are to be padded
        >>> T = 50      # Input sequence length
        >>> C = 20      # Number of classes (including blank)
        >>> N = 16      # Batch size
        >>> S = 30      # Target sequence length of longest target in batch (padding length)
        >>> S_min = 10  # Minimum target length, for demonstration purposes
        >>>
        >>> # Initialize random batch of input vectors, for *size = (T,N,C)
        >>> input = torch.randn(T, N, C).log_softmax(2).detach().requires_grad_()
        >>>
        >>> # Initialize random batch of targets (0 = blank, 1:C = classes)
        >>> target = torch.randint(low=1, high=C, size=(N, S), dtype=torch.long)
        >>>
        >>> input_lengths = torch.full(size=(N,), fill_value=T, dtype=torch.long)
        >>> target_lengths = torch.randint(low=S_min, high=S, size=(N,), dtype=torch.long)
        >>> ctc_loss = nn.CTCLoss()
        >>> loss = ctc_loss(input, target, input_lengths, target_lengths)
        >>> loss.backward()
        >>>
        >>>
        >>> # Initialize random batch of input vectors, for *size = (T,N,C)
        >>> input = torch.randn(T, N, C).log_softmax(2).detach().requires_grad_()
        >>> input_lengths = torch.full(size=(N,), fill_value=T, dtype=torch.long)
        >>>
        >>> # Initialize random batch of targets (0 = blank, 1:C = classes)
        >>> target_lengths = torch.randint(low=1, high=T, size=(N,), dtype=torch.long)
        >>> target = torch.randint(low=1, high=C, size=(sum(target_lengths),), dtype=torch.long)
        >>> ctc_loss = nn.CTCLoss()
        >>> loss = ctc_loss(input, target, input_lengths, target_lengths)
        >>> loss.backward()

    Reference:
        A. Graves et al.: Connectionist Temporal Classification:
        Labelling Unsegmented Sequence Data with Recurrent Neural Networks:
        https://www.cs.toronto.edu/~graves/icml_2006.pdf

    Note:
        In order to use CuDNN, the following must be satisfied: :attr:`targets` must be
        in concatenated format, all :attr:`input_lengths` must be `T`.  :math:`blank=0`,
        :attr:`target_lengths` :math:`\leq 256`, the integer arguments must be of
        dtype :attr:`torch.int32`.

        The regular implementation uses the (more common in PyTorch) `torch.long` dtype.


    Note:
        In some circumstances when using the CUDA backend with CuDNN, this operator
        may select a nondeterministic algorithm to increase performance. If this is
        undesirable, you can try to make the operation deterministic (potentially at
        a performance cost) by setting ``torch.backends.cudnn.deterministic =
        True``.
        Please see the notes on :doc:`/notes/randomness` for background.
    """
    __constants__ = ['blank', 'reduction']
    blank: int
    zero_infinity: bool

    def __init__(self, blank: int = 0, reduction: str = 'mean', zero_infinity: bool = False):
        super(My_CTCLoss, self).__init__(reduction=reduction)
        self.blank = blank
        self.zero_infinity = zero_infinity

    def forward(self, log_probs: torch.Tensor, targets: torch.Tensor, input_lengths: torch.Tensor, target_lengths: torch.Tensor) -> torch.Tensor:
        return functional.ctc_loss(log_probs, targets, input_lengths, target_lengths, self.blank, self.reduction,
                          self.zero_infinity)

class WER_loss(Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, preds,
            preds_sizes,
            targets,
            target_sizes):
        split_targets = []
        offset = 0
        for size in target_sizes:
            split_targets.append(targets[offset:offset + size])
            offset += size
        decoded_output, _ = self.decoder.decode(preds, preds_sizes)
        
        target_strings = self.target_decoder.convert_to_strings(split_targets)
        
        
        for x in range(len(target_strings)):
            transcript, reference = decoded_output[x][0], target_strings[x][0]

            self.calculate_metric(
                transcript=transcript,
                reference=reference
            )
        return torch.mean(torch.pow((x - y), 2))



class ErrorRate(Metric, ABC):
    def __init__(self,
                 decoder: Decoder,
                 target_decoder: GreedyDecoder,
                 save_output: bool = False,
                 dist_sync_on_step: bool = False):
        super().__init__(dist_sync_on_step=dist_sync_on_step)
        self.decoder = decoder
        self.target_decoder = target_decoder
        self.save_output = save_output

    @abstractmethod
    def calculate_metric(self, transcript, reference):
        raise NotImplementedError

    def update(self, preds: torch.Tensor,
               preds_sizes: torch.Tensor,
               targets: torch.Tensor,
               target_sizes: torch.Tensor):
        # unflatten targets
        split_targets = []
        offset = 0
        for size in target_sizes:
            split_targets.append(targets[offset:offset + size])
            offset += size
        decoded_output, _ = self.decoder.decode(preds, preds_sizes)
        
        target_strings = self.target_decoder.convert_to_strings(split_targets)
        
        
        for x in range(len(target_strings)):
            transcript, reference = decoded_output[x][0], target_strings[x][0]

            self.calculate_metric(
                transcript=transcript,
                reference=reference
            )


class CharErrorRate(ErrorRate):
    def __init__(self,
                 decoder: Decoder,
                 target_decoder: GreedyDecoder,
                 save_output: bool = False,
                 dist_sync_on_step: bool = False):
        super().__init__(
            decoder=decoder,
            target_decoder=target_decoder,
            save_output=save_output,
            dist_sync_on_step=dist_sync_on_step
        )
        self.decoder = decoder
        self.target_decoder = target_decoder
        self.save_output = save_output
        self.add_state("cer", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("n_chars", default=torch.tensor(0), dist_reduce_fx="sum")

    def calculate_metric(self, transcript, reference):
        cer_inst = self.cer_calc(transcript, reference)
        self.cer += cer_inst
        self.n_chars += len(reference.replace(' ', ''))

    def compute(self):
        cer = float(self.cer) / self.n_chars
        return cer.item() * 100

    def cer_calc(self, s1, s2):
        """
        Computes the Character Error Rate, defined as the edit distance.

        Arguments:
            s1 (string): space-separated sentence
            s2 (string): space-separated sentence
        """
        s1, s2, = s1.replace(' ', ''), s2.replace(' ', '')
        return Lev.distance(s1, s2)


class WordErrorRate(ErrorRate):
    def __init__(self,
                 decoder: Decoder,
                 target_decoder: GreedyDecoder,
                 save_output: bool = False,
                 dist_sync_on_step: bool = False):
        super().__init__(
            decoder=decoder,
            target_decoder=target_decoder,
            save_output=save_output,
            dist_sync_on_step=dist_sync_on_step
        )
        self.decoder = decoder
        self.target_decoder = target_decoder
        self.save_output = save_output
        self.add_state("wer", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("n_tokens", default=torch.tensor(0), dist_reduce_fx="sum")

    def calculate_metric(self, transcript, reference):
        wer_inst = self.wer_calc(transcript, reference)
        self.wer += wer_inst
        self.n_tokens += len(reference.split())

    def compute(self):
        wer = float(self.wer) / self.n_tokens
        return wer.item() * 100

    def wer_calc(self, s1, s2):
        """
        Computes the Word Error Rate, defined as the edit distance between the
        two provided sentences after tokenizing to words.
        Arguments:
            s1 (string): space-separated sentence
            s2 (string): space-separated sentence
        """

        # build mapping of words to integers
        b = set(s1.split() + s2.split())
        word2char = dict(zip(b, range(len(b))))

        # map the words to a char array (Levenshtein packages only accepts
        # strings)
        w1 = [chr(word2char[w]) for w in s1.split()]
        w2 = [chr(word2char[w]) for w in s2.split()]

        return Lev.distance(''.join(w1), ''.join(w2))


@torch.no_grad()
def run_evaluation(test_loader,
                   model,
                   decoder: Decoder,
                   device: torch.device,
                   target_decoder: Decoder,
                   precision: int):
    print(model)
    model.eval()
    loss_ = []
    wer = WordErrorRate(
        decoder=decoder,
        target_decoder=target_decoder
    )
    cer = CharErrorRate(
        decoder=decoder,
        target_decoder=target_decoder
    )
    for i, (batch) in tqdm(enumerate(test_loader), total=len(test_loader)):
        inputs, targets, input_percentages, target_sizes = batch
        print(inputs.size())
        input_sizes = input_percentages.mul_(int(inputs.size(3))).int()
        inputs = inputs.to(device)

        with autocast(enabled=precision == 16):
            out, output_sizes = model(inputs, input_sizes)
        decoded_output, _ = decoder.decode(out, output_sizes)
        criterion = CTCLoss(blank=model.labels.index('_'), reduction='mean', zero_infinity=True)
        #print(out.size(),output_sizes,target_sizes)
        wer.update(
            preds=out,
            preds_sizes=output_sizes,
            targets=targets,
            target_sizes=target_sizes
        )
        cer.update(
            preds=out,
            preds_sizes=output_sizes,
            targets=targets,
            target_sizes=target_sizes
        )
        out = out.transpose(0, 1)
        out = out.log_softmax(-1)
        loss = criterion(out, targets, output_sizes, target_sizes)
        loss_.append(loss)


    plt.plot(loss_)
    plt.xlabel('batch')
    plt.ylabel('loss')
    plt.show()
    plt.savefig("loss.png")
    return wer.compute(), cer.compute()

@torch.no_grad()
def attack_test_evaluation(input_word, target_word, delta,
                   test_loader,
                   model,
                   decoder: Decoder,
                   device: torch.device,
                   target_decoder: Decoder,
                   precision: int):

    model.eval()
    loss_ = []
    wer = WordErrorRate(
        decoder=decoder,
        target_decoder=target_decoder
    )
    cer = CharErrorRate(
        decoder=decoder,
        target_decoder=target_decoder
    )
    attack_success = 0
    attack_num = 0
    r = 0
    r_num = 0
    for i, (batch) in tqdm(enumerate(test_loader), total=len(test_loader)):
        inputs, targets, input_percentages, target_sizes = batch
        
        input_sizes = input_percentages.mul_(int(inputs.size(3))).int()
        inputs = inputs.to(device)
        delta_ = delta.repeat(inputs.size(0),1,1,1)
        inputs = inputs + delta_
        with autocast(enabled=precision == 16):
            out, output_sizes = model(inputs, input_sizes)
        decoded_output, _ = decoder.decode(out, output_sizes)
        split_targets = []
        offset = 0
        for size in target_sizes:
            split_targets.append(targets[offset:offset + size])
            offset += size
        true_output = target_decoder.convert_to_strings(split_targets)
        
        for i in range(len(true_output)):
            if true_output[i][0] == input_word:
                
                attack_num+=1
                attack_success += (decoded_output[i][0]==target_word)
            else:
                r_num+=1
                
                r +=  (true_output[i][0] == decoded_output[i][0])
                



        criterion = CTCLoss(blank=model.labels.index('_'), reduction='mean', zero_infinity=True)


        out = out.transpose(0, 1)
        out = out.log_softmax(-1)
        loss = criterion(out, targets, output_sizes, target_sizes)
        loss_.append(loss)
        if (decoded_output[0][0] == 'LEFT'):
            print(loss)
    print("attack_succes_rate", attack_success/attack_num)
    print("other_word_translate_rate", r/r_num)
    plt.plot(loss_)
    plt.xlabel('batch')
    plt.ylabel('loss')
    plt.show()
    plt.savefig("loss.png")
    return wer.compute(), cer.compute()


def set_bn_eval(m):
    classname = m.__class__.__name__
    print(classname)
    if classname.find('batch_norm') != -1:
        #print("batch_norm")
        m.eval()

def train_evaluation(input_word, target_word, attack_length,
                   test_loader,
                   model,
                   decoder: Decoder,
                   device: torch.device,
                   target_decoder: Decoder,
                   precision: int):
    
    
    criterion = CTCLoss(blank=model.labels.index('_'), reduction='mean', zero_infinity=True)
    model.train()
    wer = WordErrorRate(
        decoder=decoder,
        target_decoder=target_decoder
    )
    cer = CharErrorRate(
        decoder=decoder,
        target_decoder=target_decoder
    )
    
    delta = None
    loss = 0
    loss_ = []
    for i, (batch) in tqdm(enumerate(test_loader), total=len(test_loader)):

        inputs, targets, input_percentages, target_sizes = batch
        
        if(i==0):
            delta = torch.rand(1,1,inputs.size(2), attack_length) * 0.1
            delta = delta.to(device)
            delta.requires_grad = True
            optimizer = optim.Adam([delta])
            print(delta.size())   
        
        #delta = WaveNet(1,1,32,32,10,4,4)
        delta_ = delta.repeat(inputs.size(0),1,1,1)
        delta_ = delta_.to("cpu")
        delta_ = torch.cat((delta_,torch.zeros(inputs.size(0),1,inputs.size(2),1)),3)
        
        for idx in range(0,inputs.size(3)-delta_.size(3),delta_.size(3)):
            inputs[:,:,:,idx:idx+delta_.size(3)] += delta_
        inputs = inputs.to(device)
        input_sizes = input_percentages.mul_(int(inputs.size(3))).int()

        
        
        with autocast(enabled=precision == 16):
            out, output_sizes = model(inputs, input_sizes)
        
        decoded_output, _ = decoder.decode(out, output_sizes)
        

        '''
        split_targets = []
        offset = 0
        for size in target_sizes:
            split_targets.append(targets[offset:offset + size])
            offset += size
        true_output = target_decoder.convert_to_strings(split_targets)
        print(targets)
        print(output_sizes)
        train_word = ''
        for i in range(len(true_output)):
            each_word = true_output[i]
            decoded_output_i = decoded_output[i]
            decoded_output_i = decoded_output_i[0]
            
            if each_word[0] == input_word:
                train_word = target_word
            else:
                train_word = each_word[0]
        '''

        
        
        wer.update(
            preds=out,
            preds_sizes=output_sizes,
            targets=targets,
            target_sizes=target_sizes
        )
        cer.update(
            preds=out,
            preds_sizes=output_sizes,
            targets=targets,
            target_sizes=target_sizes
        )
        print(len(out[0]),len(targets))
        out = out.transpose(0, 1)
        out = out.log_softmax(-1)
        loss = criterion(out, targets, output_sizes, target_sizes)
        
        loss += 4 * delta.abs().mean()
        loss_.append(loss)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        #delta = torch.clamp(delta-grad[0] * 0.02, min=-0.2, max=0.2)
    delta = delta.detach().to('cpu').numpy().reshape(delta.size(2),delta.size(3))
    plt.plot(loss_)
    plt.xlabel('batch')
    plt.ylabel('loss')
    plt.show()
    plt.savefig("loss.png")
    plt.close()
    loss_batch = []
    for i in range(0,len(loss_)-20,20):
        average_loss = sum(loss_[i:i+20])/20
        loss_batch.append(average_loss)
    plt.plot(loss_batch)
    plt.xlabel('batch')
    plt.ylabel('loss')
    plt.show()
    plt.savefig("loss_.png")
    plt.close()
    print(delta)
    numpy.savetxt("delta.txt",delta)
    numpy.save("delta.npy",delta)
    return wer.compute(), cer.compute()
