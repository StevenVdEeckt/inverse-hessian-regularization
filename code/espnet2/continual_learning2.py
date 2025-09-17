# Copyright 2025 Steven Vander Eeckt - KU Leuven
# Continual Learning extensions for End-to-End ASR (ESPnet2)

# imports
import numpy as np
import torch
import logging
from abc import ABC, abstractmethod
import typing
import sentencepiece as spm
import random
import copy

# espnet imports
from espnet2.torch_utils.model_summary import model_summary
from espnet2.fileio.read_text import load_num_sequence_text

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


#####################################################################################################################
########################### HELP FUNCTIONS ##########################################################################
#####################################################################################################################
def _recursive_to(
        xs,
        device
):
    if torch.is_tensor(xs):
        return xs.to(device)
    if isinstance(xs, tuple) or isinstance(xs, list):
        return tuple(_recursive_to(x, device) for x in xs)
    if isinstance(xs, dict):
        return {n: _recursive_to(p, device) for n, p in xs.items()}
    return xs

def copy_model(
        model: torch.nn.Module,
        device: str,
        train: bool = False,
        requires_grad: bool = False,
):
    """
        Copy the model and make it untrainable

        :param torch.nn.Module model: the model
        :param str device:  cpu or gpu device to map model to
        :param bool train: if False, model is set to eval mode
        :param bool requires_grad:
    """
    # for batch norm and dropout layers
    old_model = copy.deepcopy(model).train(train)
    for name, param in old_model.named_parameters():
        param.requires_grad = requires_grad
    return old_model.to(device)

def item(t):
    try:
        return t.item()
    except:
        return t



#####################################################################################################################
########################### MEMORY CLASSES ##########################################################################
#####################################################################################################################


class Memory(ABC):

    @abstractmethod
    def update(
            self,
            xs,
    ):
        pass

    @abstractmethod
    def next(
            self
    ):
        pass

    @abstractmethod
    def save(
            self,
            outdir: str,
    ):
        pass

    @staticmethod
    def load(
            filename: str,
    ):
        pass

    @staticmethod
    def get_unique_key(
            names: typing.List[str],
    ):
        """
            Generates unique key for batch with utterances in names

            :param List[str] names: list of utterance IDs of batch
        """
        return "_".join(sorted(names))
    
    @abstractmethod
    def from_original_memory(
            self,
            names,
    ):
        pass



#####################################################################################################################
########################### ABSTRACT CLASSES ########################################################################
#####################################################################################################################


class CLMethod(ABC):
    """
        CL Method Abstract Class
    """
    pass



class LossBasedCL(CLMethod):
    """
        Loss-Based CL Abstract Class

        Includes all methods, regularization and rehearsal, which enable CL by means of a regularizing loss

        :method compute_loss: each class inheriting LossBasedCL must have a compute_loss(model) method
    """
    @abstractmethod
    def compute_loss(
            self,
            model: torch.nn.Module,
    ):
        """
            Computes the loss given the model
            :param torch.nn.Module model:
        """
        pass


class WeightBasedCL(CLMethod):
    """
        Weight-based CL Abstract Class

        Includes all methods, regularization and rehearsal, which update the weights (after an epoch or iteration)

        :method update_model
    """
    def __init__(self):
        self.epoch = 0

    def init_model(
            self,
            epoch: int,
    ):
        self.epoch = epoch

    @abstractmethod
    def update_model(
            self,
            model: torch.nn.Module
    ):
        """
            Called after each epoch to update the weights of the model
        """
        pass

    @staticmethod
    def to_param(
            x: torch.tensor,
            sizes,
    ):
        """
            Turns a vector into a state_dict
            :param torch.tensor x: the tensor as 1D vector
            :param dict sizes: dict with key the layer and value a dict with p.numel() and p.size()

        """
        sd, k = {}, 0
        for n, p in sizes.items():
            sd[n] = x[k:k + p['numel']].view(p['size'])
            k += p['numel']
        return sd


class Rehearsal(CLMethod):
    """
        Rehearsal Abstract Class
        Generic class to be inherited by Rehearsal-based CL methods

        :param str device:
        :param str task_file : task file to map utterances to tasks (task IDs)
        :param str task_order : sets the task IDs in the right order (e.g. '1 2 0 3' means that task with task ID = 1 is
                                the first task, task with task ID = 2 the second, and so on)
    """
    def __init__(
            self,
            device: str,
            task_file: str = "",
            task_order: str = "",
            group: bool = False,
            memory_text: str = "",
            num_tokens: int = 5000,
    ):
        self.device = device
        # check if model has task specific part
        if group:
            assert num_tokens > 0
            assert memory_text is not None
            self.get_sp = lambda lang: spm.SentencePieceProcessor(model_file=f'data/{lang}_token_list/bpe_unigram{num_tokens}/bpe.model')
            self.other_sp = {}
            self.utt2text_ids = {}
            self.utt2text = self._prepare_utt2text(memory_text)
            self.get_batch = lambda return_names=False: self._get_batch_and_lang(return_names)
        elif task_file and task_order:
            # load utterance to task mapping
            logging.info("opening task file.. %s" % task_file)
            utt2task = load_num_sequence_text(task_file, loader_type="csv_int")
            assert task_file != ""
            # load task ID ordering
            task_order = {int(task): i for i, task in enumerate(task_order.split(" "), 0)}
            # dicts to map utterance and batches to tasks
            self.utt2task = {utt: task_order[task[0]] for utt, task in utt2task.items()}
            self.batch2task = {}
            # set the default get_batch() function.
            self.get_batch = lambda return_names=False: self._get_batch_and_task(return_names)
        else:
            # set the default get_batch() function.
            self.get_batch = lambda return_names=False: self._get_batch(return_names)

    def set_loader(
            self,
            loader
    ):
        """
            Set the loader and iterator of the rehearsal-based method

            :param torch.Dataloader loader: the dataloader for the memory of the rehearsal-based method
        """
        self.loader = loader.build_iter(epoch=0, shuffle=True)
        self.iter = iter(self.loader)

    @staticmethod
    def get_unique_key(
            names: typing.List[str],
    ):
        """
            Generates unique key for batch with utterances in names

            :param List[str] names: list of utterance IDs of batch
        """
        return "_".join(sorted(names))

    def _get_batch(
            self,
            return_names: bool = False
    ):
        """
            Sample a batch from the iterator (memory)
        """
        try:
            #names, xs = self.iter.next()
            names, xs = next(self.iter)
        except:  # if at the end of the iterator
            self.iter = iter(self.loader)
            #names, xs = self.iter.next()
            names, xs = next(self.iter)
        if return_names:
            return _recursive_to(xs, self.device), names
        return  _recursive_to(xs, self.device)

    def _get_batch_and_lang(
            self,
            return_names: bool = False,
    ):
        """
            Sample a batch from the iterator (memory)
        """
        try:
            names, xs = next(self.iter)
        except:  # if at the end of the iterator
            self.iter = iter(self.loader)
            names, xs = next(self.iter)
        names, xs = self._convert_batch(names, xs)
        if return_names:
            return _recursive_to(xs, self.device), names
        return  _recursive_to(xs, self.device)

    def _convert_batch(
            self,
            names: typing.List[str],
            batch: typing.Dict,
    ):
        key = self.get_unique_key(names)
        # if utterance has already been converted once
        try:
            lang, _ = self._get_lang_and_sp_model(names)
            text, text_lengths = self.utt2text_ids[key]
        except KeyError as e:
            # get lang and corresponding sp model
            old_text, old_text_lengths = batch['text'], batch['text_lengths']
            # get text and text_lengths in old alphabet
            text, text_lengths = old_text.clone(), old_text_lengths.clone()
            # get lang and sp_model of new alphabet
            lang, sp_model = self._get_lang_and_sp_model(names)
            # convert text
            new_text, new_text_lengths = [], []
            for i in range(old_text.size(0)):
                # convert text to sequence for correct alphabet
                t = sp_model.encode(self.utt2text[names[i]], out_type=int)
                # substract 1 (because we added 1s before)
                t = [j - 1 for j in t]
                # add substract 1 and append -1s
                t_ = torch.tensor(t, device=self.device)
                # replace text[i] by t_
                new_text.append(t_)
                # update text_lengths
                new_text_lengths.append(len(t))
            # Determine max length
            max_len = max(new_text_lengths)
            pad_idx = -1
            padded_texts = torch.full((len(new_text), max_len), fill_value=pad_idx, device=self.device)
            for i, t in enumerate(new_text):
                padded_texts[i, :t.size(0)] = t
            # Set final text and length tensors
            text = padded_texts
            text_lengths = torch.tensor(new_text_lengths, dtype=torch.long, device=self.device)
            # store in utt2text
            self.utt2text_ids[key] = (text, text_lengths)
        batch['text'], batch['text_lengths'], batch['lang_sym'] = text, text_lengths, lang
        return names, batch

    def _get_lang_and_sp_model(
            self,
            names: typing.List[str],
    ):
        def get_lang(utt_key):
            speaker_id = utt_key.split("-")[1]
            lang = speaker_id.split("_")[2]
            return lang
        langs = [get_lang(name) for name in names]
        assert len(list(set(langs))) == 1, f"Found more than one language: {list(set(langs))}"
        lang = langs[0]
        if not lang in self.other_sp.keys():
            self.other_sp[lang] = self.get_sp(lang)
        return lang, self.other_sp[lang]

    @staticmethod
    def _prepare_utt2text(
            memory_text: str,
    ):
        # read memory text and store in utt2gt (ground truth)
        utt2gt = {}
        with open(memory_text, "r", encoding="utf-8") as f:
            line = f.readline()
            while line:
                line = line.strip("\n")
                utt, text = line.split(" ")[0], " ".join(line.split(" ")[1:])
                utt2gt[utt] = text
                line = f.readline()
        return utt2gt

    def get_task_from_batch(
            self,
            names: typing.List[str]
    ):
        """
            Given (the utterance IDs of a batch), return the task number

            :param List[str] names: list containing the utterance IDs of the batch
        """
        # generate a unique hashable key
        key = self.get_unique_key(names)
        # select task from batch2task if it exists:
        try:
            task = self.batch2task[key]
            return task
        except Exception as e:  # the first time we encounter this batch
            tasks = set([self.utt2task[name] for name in names])
            assert len(tasks) == 1, "batch %s had multiple tasks: %s" % (str(names), str(tasks))
            self.batch2task[key] = list(tasks)[0]
            return list(tasks)[0]

    def _get_batch_and_task(
            self,
            return_names: bool = False
    ):
        """
            Sample a batch from the iterator (memory)
            and add the task to the batch.

            :param bool return_names: return utterance IDs
        """
        try:
            names, xs = self.iter.next()
        except:  # if at the end of the iterator
            self.iter = iter(self.loader)
            names, xs = self.iter.next()
        # get the task of the batch
        task = self.get_task_from_batch(names)
        # add it to the batch
        xs['task_label'] = task
        if return_names:
            return _recursive_to(xs, self.device), names
        return _recursive_to(xs, self.device)




#####################################################################################################################
########################### REHEARSAL-BASED CL METHODS ##############################################################
#####################################################################################################################

class ER(LossBasedCL, Rehearsal):
    """
        Experience Replay extension for ASR model.

        :param str device: the device for the model and data
        :param float alpha: the weight of the regularization
    """
    def __init__(
            self,
            device: str,
            alpha: float = 1,
            task_file: str = "",
            task_order: str = "",
            memory_text: str = "",
            num_tokens: int = 5000,
            group: bool = False,
    ):
        super(ER, self).__init__(device=device, task_file=task_file, task_order=task_order,
                                 memory_text=memory_text, num_tokens=num_tokens, group=group)
        self.alpha = alpha
        logging.info("CL Method: ER with alpha=%.2f" % (alpha))

    def compute_loss(
            self,
            model: torch.nn.Module
    ):
        """
            Computes the ER loss

            :param torch.nn.Module model: the current model
        """
        # sample a batch
        batch = self.get_batch()
        # compute loss on new model
        loss = model.forward_er(**batch)
        # multiply by regularization weight
        return self.alpha * loss

#####################################################################################################################
########################### REGULARIZATION-BASED CL METHODS #########################################################
#####################################################################################################################

class UOE(CLMethod):
    def __init__(
            self,
            device: str,
            **kwargs,
    ):
        self.device = device

    def set_model(
            self,
            model: torch.nn.Module,
    ):
        for module in model.modules():
            if isinstance(module, torch.nn.LayerNorm):
                module.weight.requires_grad_(False)
                module.bias.requires_grad_(False)
        for name, param in model.named_parameters():
            if not 'encoders' in name:
                param.requires_grad_(False)
        logging.info(model_summary(model))


class CLRLTuning(WeightBasedCL):
    def __init__(
            self,
            device: str,
            K=1
    ):
        super().__init__()
        self.device = device
        self.K = K
        self.num_encs = 0

    def get_encoders_to_unfreeze(self):
        selected_encoders = random.sample(range(self.num_encs), self.K)
        logging.info(f"Freezing encoders {sorted(selected_encoders)}")
        names = [f'encoder.encoders.{i}.' for i in selected_encoders]
        def freeze(layer_name):
            for name in names:
                if name in layer_name:
                    return False
            return True
        return freeze

    def set_model(
            self,
            model: torch.nn.Module,
    ):
        # set the number of encoders
        self.num_encs = len(model.encoder.encoders)
        # first set to trainable all parameters
        for name, param in model.named_parameters():
            param.requires_grad_(True)
        # freeze decoder
        for name, param in model.named_parameters():
            if not 'encoders' in name:
                param.requires_grad_(False)
        # freeze all except K encoders
        freeze_func = self.get_encoders_to_unfreeze()
        for name, param in model.named_parameters():
            if freeze_func(name):
                param.requires_grad_(False)
        logging.info(model_summary(model))

    def update_model(
            self,
            model: torch.nn.Module
    ):
        """
            Called after each epoch to update the weights of the model
        """
        # freeze all except K encoders
        freeze_func = self.get_encoders_to_unfreeze()
        for name, param in model.named_parameters():
            if freeze_func(name):
                param.requires_grad_(False)
            else:
                param.requires_grad_(True)
        logging.info(model_summary(model))
