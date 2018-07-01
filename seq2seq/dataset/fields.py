import logging
import torchtext
import torch

class SourceField(torchtext.data.Field):
    """ Wrapper class of torchtext.data.Field that forces batch_first and include_lengths to be True. """

    def __init__(self, **kwargs):
        logger = logging.getLogger(__name__)

        if kwargs.get('batch_first') is False:
            logger.warning("Option batch_first has to be set to use pytorch-seq2seq.  Changed to True.")
        kwargs['batch_first'] = True
        if kwargs.get('include_lengths') is False:
            logger.warning("Option include_lengths has to be set to use pytorch-seq2seq.  Changed to True.")
        kwargs['include_lengths'] = True

        super(SourceField, self).__init__(**kwargs)

class AuxSourceField(torchtext.data.Field):
    """ Wrapper class of torchtext.data.Field that 
    forces batch_first and include_lengths to be True. and support lex info data """

    def __init__(self, lexdata, **kwargs):
        logger = logging.getLogger(__name__)

        if kwargs.get('batch_first') is False:
            logger.warning("Option batch_first has to be set to use pytorch-seq2seq.  Changed to True.")
        kwargs['batch_first'] = True
        if kwargs.get('include_lengths') is False:
            logger.warning("Option include_lengths has to be set to use pytorch-seq2seq.  Changed to True.")
        kwargs['include_lengths'] = True
        self.lexdata = lexdata

        super(AuxSourceField, self).__init__(**kwargs)
    
    def getLexVariables(self, batch, device=None):
        vocab = self.vocab
        batch_lexvars = []
        for ex in batch:
            words = [vocab.itos[x] for x in ex]
            lex_vars = []
            for word_x in words:
                aux_vec = [0,0,0,0]
                # aux_vec :: [age, is_neg, is_neu, is_pos]
                lex_x = self.lexdata.get(word_x, {})
                age = lex_x.get("age")
                pol_idx = lex_x.get("polarity")
                if age:
                    aux_vec[0] = age
                if pol_idx:
                    aux_vec[pol_idx] = 1
                lex_vars.append(aux_vec)
            batch_lexvars.append(lex_vars)
        tensor_lexvars = torch.tensor(batch_lexvars, dtype=torch.float, device=device)
        return tensor_lexvars

class TargetField(torchtext.data.Field):
    """ Wrapper class of torchtext.data.Field that forces batch_first to be True and prepend <sos> and append <eos> to sequences in preprocessing step.

    Attributes:
        sos_id: index of the start of sentence symbol
        eos_id: index of the end of sentence symbol
    """

    SYM_SOS = '<sos>'
    SYM_EOS = '<eos>'

    def __init__(self, **kwargs):
        logger = logging.getLogger(__name__)

        if kwargs.get('batch_first') == False:
            logger.warning("Option batch_first has to be set to use pytorch-seq2seq.  Changed to True.")
        kwargs['batch_first'] = True
        if kwargs.get('preprocessing') is None:
            kwargs['preprocessing'] = lambda seq: [self.SYM_SOS] + seq + [self.SYM_EOS]
        else:
            func = kwargs['preprocessing']
            kwargs['preprocessing'] = lambda seq: [self.SYM_SOS] + func(seq) + [self.SYM_EOS]

        self.sos_id = None
        self.eos_id = None
        super(TargetField, self).__init__(**kwargs)

    def build_vocab(self, *args, **kwargs):
        super(TargetField, self).build_vocab(*args, **kwargs)
        self.sos_id = self.vocab.stoi[self.SYM_SOS]
        self.eos_id = self.vocab.stoi[self.SYM_EOS]
