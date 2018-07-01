import torch
from torch.autograd import Variable


class PredictorMInput(object):

    def __init__(self, model, src_vocab, tgt_vocab):
        """
        Predictor class to evaluate for a given model.
        Args:
            model (seq2seq.models): trained model. This can be loaded from a checkpoint
                using `seq2seq.util.checkpoint.load`
            src_vocab (seq2seq.dataset.vocabulary.Vocabulary): source sequence vocabulary
            tgt_vocab (seq2seq.dataset.vocabulary.Vocabulary): target sequence vocabulary
        """
        if torch.cuda.is_available():
            self.model = model.cuda()
        else:
            self.model = model.cpu()
        self.model.eval()
        self.src_vocab = src_vocab
        self.tgt_vocab = tgt_vocab

    def getLexVariables(self, batch, vocab, lexdata):
        batch_lexvars = []
        for ex in batch:
            words = [vocab.itos[x] for x in ex]
            lex_vars = []
            for word_x in words:
                aux_vec = [0,0,0,0]
                # aux_vec :: [age, is_neg, is_neu, is_pos]
                lex_x = lexdata.get(word_x, {})
                age = lex_x.get("age")
                pol_idx = lex_x.get("polarity")
                if age:
                    aux_vec[0] = age
                if pol_idx:
                    aux_vec[pol_idx] = 1
                lex_vars.append(aux_vec)
            batch_lexvars.append(lex_vars)
        tensor_lexvars = torch.tensor(batch_lexvars, dtype=torch.float)
        if torch.cuda.is_available():
            tensor_lexvars = tensor_lexvars.cuda()
        return tensor_lexvars

    def get_decoder_features(self, src_seq, lexdata):
        batch = [self.src_vocab.stoi[tok] for tok in src_seq]
        src_id_seq = torch.LongTensor(batch).view(1, -1)
        if torch.cuda.is_available():
            src_id_seq = src_id_seq.cuda()
        src_lex_vec = self.getLexVariables([batch], self.src_vocab, lexdata)
        with torch.no_grad():
            softmax_list, _, other = self.model(src_id_seq, src_lex_vec, [len(src_seq)])

        return other

    def predict(self, src_seq, lexdata):
        """ Make prediction given `src_seq` as input.

        Args:
            src_seq (list): list of tokens in source language

        Returns:
            tgt_seq (list): list of tokens in target language as predicted
            by the pre-trained model
        """
        other = self.get_decoder_features(src_seq, lexdata)

        length = other['length'][0]

        tgt_id_seq = [other['sequence'][di][0].data[0] for di in range(length)]
        tgt_seq = [self.tgt_vocab.itos[tok] for tok in tgt_id_seq]
        return tgt_seq

    def predict_n(self, src_seq, lexdata, n=1):
        """ Make 'n' predictions given `src_seq` as input.

        Args:
            src_seq (list): list of tokens in source language
            n (int): number of predicted seqs to return. If None,
                     it will return just one seq.

        Returns:
            tgt_seq (list): list of tokens in target language as predicted
                            by the pre-trained model
        """
        other = self.get_decoder_features(src_seq, lexdata)

        result = []
        for x in range(0, int(n)):
            length = other['topk_length'][0][x]
            tgt_id_seq = [other['topk_sequence'][di][0, x, 0].data[0] for di in range(length)]
            tgt_seq = [self.tgt_vocab.itos[tok] for tok in tgt_id_seq]
            result.append(tgt_seq)

        return result
