# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HugginFace Inc. team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""PyTorch BERT model."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy
import json
import math
import six
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss
from six import string_types
from common_decanlp import LSTMDecoderAttention, LSTMDecoder, Feedforward, mask, TransformerDecoder, positional_encodings_like, LayerNorm
# from tokenization import load_idx_to_token

PAD_INDEX = 0

def gelu(x):
    """Implementation of the gelu activation function.
        For information: OpenAI GPT's gelu is slightly different (and gives slightly different results):
        0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))
    """
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))


def swish(x):
    return x * torch.sigmoid(x)


ACT2FN = {"gelu": gelu, "relu": torch.nn.functional.relu, "swish": swish}


class BertConfig(object):
    """Configuration class to store the configuration of a `BertModel`.
    """
    def __init__(self,
                vocab_size,
                hidden_size=768,
                num_hidden_layers=12,
                num_attention_heads=12,
                intermediate_size=3072,
                hidden_act="gelu",
                hidden_dropout_prob=0.1,
                attention_probs_dropout_prob=0.1,
                max_position_embeddings=512,
                type_vocab_size=16,
                initializer_range=0.02):
        """Constructs BertConfig.

        Args:
            vocab_size: Vocabulary size of `inputs_ids` in `BertModel`.
            hidden_size: Size of the encoder layers and the pooler layer.
            num_hidden_layers: Number of hidden layers in the Transformer encoder.
            num_attention_heads: Number of attention heads for each attention layer in
                the Transformer encoder.
            intermediate_size: The size of the "intermediate" (i.e., feed-forward)
                layer in the Transformer encoder.
            hidden_act: The non-linear activation function (function or string) in the
                encoder and pooler. If string, "gelu", "relu" and "swish" are supported.
            hidden_dropout_prob: The dropout probabilitiy for all fully connected
                layers in the embeddings, encoder, and pooler.
            attention_probs_dropout_prob: The dropout ratio for the attention
                probabilities.
            max_position_embeddings: The maximum sequence length that this model might
                ever be used with. Typically set this to something large just in case
                (e.g., 512 or 1024 or 2048).
            type_vocab_size: The vocabulary size of the `token_type_ids` passed into
                `BertModel`.
            initializer_range: The sttdev of the truncated_normal_initializer for
                initializing all weight matrices.
        """
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.hidden_act = hidden_act
        self.intermediate_size = intermediate_size
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.max_position_embeddings = max_position_embeddings
        self.type_vocab_size = type_vocab_size
        self.initializer_range = initializer_range

    @classmethod
    def from_dict(cls, json_object):
        """Constructs a `BertConfig` from a Python dictionary of parameters."""
        config = BertConfig(vocab_size=None)
        for (key, value) in six.iteritems(json_object):
            config.__dict__[key] = value
        return config

    @classmethod
    def from_json_file(cls, json_file):
        """Constructs a `BertConfig` from a json file of parameters."""
        with open(json_file, "r") as reader:
            text = reader.read()
        return cls.from_dict(json.loads(text))

    def to_dict(self):
        """Serializes this instance to a Python dictionary."""
        output = copy.deepcopy(self.__dict__)
        return output

    def to_json_string(self):
        """Serializes this instance to a JSON string."""
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"


class BERTLayerNorm(nn.Module):
    def __init__(self, config, variance_epsilon=1e-12):
        """Construct a layernorm module in the TF style (epsilon inside the square root).
        """
        super(BERTLayerNorm, self).__init__()
        self.gamma = nn.Parameter(torch.ones(config.hidden_size))
        self.beta = nn.Parameter(torch.zeros(config.hidden_size))
        self.variance_epsilon = variance_epsilon

    def forward(self, x):
        u = x.mean(-1, keepdim=True)
        s = (x - u).pow(2).mean(-1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.variance_epsilon)
        return self.gamma * x + self.beta

class BERTEmbeddings(nn.Module):
    def __init__(self, config):
        super(BERTEmbeddings, self).__init__()
        """Construct the embedding module from word, position and token_type embeddings.
        """
        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size)
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        self.token_type_embeddings = nn.Embedding(config.type_vocab_size, config.hidden_size)

        # self.LayerNorm is not snake-cased to stick with TensorFlow model variable name and be able to load
        # any TensorFlow checkpoint file
        self.LayerNorm = BERTLayerNorm(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, input_ids, token_type_ids=None, position=None):
        seq_length = input_ids.size(1)
        if position is None:
            position_ids = torch.arange(seq_length, dtype=torch.long, device=input_ids.device)
            position_ids = position_ids.unsqueeze(0).expand_as(input_ids)
        else:
            position_ids = torch.tensor([position], dtype=torch.long, device=input_ids.device)
            position_ids = position_ids.unsqueeze(0).expand_as(input_ids)
        
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)

        words_embeddings = self.word_embeddings(input_ids)
        position_embeddings = self.position_embeddings(position_ids)
        token_type_embeddings = self.token_type_embeddings(token_type_ids)
        
        # if position is None:
        #     print('inputs_ids is \n', input_ids[:, :3])
        #     print('words_embeddings is \n', words_embeddings[:, :3, :5])
        #     print('position_ids is \n', position_ids[:, :3])
        #     print('position_embeddings is \n', position_embeddings[:, :3, :5])
        #     print('token_type_ids is \n', token_type_ids[:, :3])
        #     print('token_type_embeddings is \n', token_type_embeddings[:, :3, :5])

        embeddings = words_embeddings + position_embeddings + token_type_embeddings
        # if position is None:
        #     print('embeddings is \n', embeddings[:, :3, :5])
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        # if position is None:
        #     print('embeddings after ln&dp is \n', embeddings[:, :3, :5])
        return embeddings


class BERTSelfAttention(nn.Module):
    def __init__(self, config):
        super(BERTSelfAttention, self).__init__()
        if config.hidden_size % config.num_attention_heads != 0:
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (config.hidden_size, config.num_attention_heads))
        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)

        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states, attention_mask):
        mixed_query_layer = self.query(hidden_states)
        mixed_key_layer = self.key(hidden_states)
        mixed_value_layer = self.value(hidden_states)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        # Apply the attention mask is (precomputed for all layers in BertModel forward() function)
        attention_scores = attention_scores + attention_mask

        # Normalize the attention scores to probabilities.
        attention_probs = nn.Softmax(dim=-1)(attention_scores)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)

        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        return context_layer


class BERTSelfOutput(nn.Module):
    def __init__(self, config):
        super(BERTSelfOutput, self).__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.LayerNorm = BERTLayerNorm(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class BERTAttention(nn.Module):
    def __init__(self, config):
        super(BERTAttention, self).__init__()
        self.self = BERTSelfAttention(config)
        self.output = BERTSelfOutput(config)

    def forward(self, input_tensor, attention_mask):
        self_output = self.self(input_tensor, attention_mask)
        attention_output = self.output(self_output, input_tensor)
        return attention_output


class BERTIntermediate(nn.Module):
    def __init__(self, config):
        super(BERTIntermediate, self).__init__()
        self.dense = nn.Linear(config.hidden_size, config.intermediate_size)
        self.intermediate_act_fn = ACT2FN[config.hidden_act] \
            if isinstance(config.hidden_act, string_types) else config.hidden_act

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)
        return hidden_states


class BERTOutput(nn.Module):
    def __init__(self, config):
        super(BERTOutput, self).__init__()
        self.dense = nn.Linear(config.intermediate_size, config.hidden_size)
        self.LayerNorm = BERTLayerNorm(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class BERTLayer(nn.Module):
    def __init__(self, config):
        super(BERTLayer, self).__init__()
        self.attention = BERTAttention(config)
        self.intermediate = BERTIntermediate(config)
        self.output = BERTOutput(config)

    def forward(self, hidden_states, attention_mask):
        attention_output = self.attention(hidden_states, attention_mask)
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)
        return layer_output


class BERTEncoder(nn.Module):
    def __init__(self, config):
        super(BERTEncoder, self).__init__()
        layer = BERTLayer(config)
        self.layer = nn.ModuleList([copy.deepcopy(layer) for _ in range(config.num_hidden_layers)])    

    def forward(self, hidden_states, attention_mask):
        all_encoder_layers = []
        for layer_module in self.layer:
            hidden_states = layer_module(hidden_states, attention_mask)
            all_encoder_layers.append(hidden_states)
        return all_encoder_layers


class BERTPooler(nn.Module):
    def __init__(self, config):
        super(BERTPooler, self).__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.activation = nn.Tanh()

    def forward(self, hidden_states):
        # We "pool" the model by simply taking the hidden state corresponding
        # to the first token.
        first_token_tensor = hidden_states[:, 0]
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        return pooled_output


class BertModel(nn.Module):
    """BERT model ("Bidirectional Embedding Representations from a Transformer").

    Example usage:
    ```python
    # Already been converted into WordPiece token ids
    input_ids = torch.LongTensor([[31, 51, 99], [15, 5, 0]])
    input_mask = torch.LongTensor([[1, 1, 1], [1, 1, 0]])
    token_type_ids = torch.LongTensor([[0, 0, 1], [0, 2, 0]])

    config = modeling.BertConfig(vocab_size=32000, hidden_size=512,
        num_hidden_layers=8, num_attention_heads=6, intermediate_size=1024)

    model = modeling.BertModel(config=config)
    all_encoder_layers, pooled_output = model(input_ids, token_type_ids, input_mask)
    ```
    """
    def __init__(self, config: BertConfig):
        """Constructor for BertModel.

        Args:
            config: `BertConfig` instance.
        """
        super(BertModel, self).__init__()
        self.embeddings = BERTEmbeddings(config)
        self.encoder = BERTEncoder(config)
        self.pooler = BERTPooler(config)

    def forward(self, input_ids, token_type_ids=None, attention_mask=None):
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)

        # We create a 3D attention mask from a 2D tensor mask.
        # Sizes are [batch_size, 1, 1, to_seq_length]
        # So we can broadcast to [batch_size, num_heads, from_seq_length, to_seq_length]
        # this attention mask is more simple than the triangular masking of causal attention
        # used in OpenAI GPT, we just need to prepare the broadcast dimension here.
        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)

        # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
        # masked positions, this operation will create a tensor which is 0.0 for
        # positions we want to attend and -10000.0 for masked positions.
        # Since we are adding it to the raw scores before the softmax, this is
        # effectively the same as removing these entirely.
        extended_attention_mask = extended_attention_mask.to(dtype=next(self.parameters()).dtype) # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -1000000.0

        # print('extended_attention_mask:\n',extended_attention_mask)
        embedding_output = self.embeddings(input_ids, token_type_ids)
        # print('embedding_output:\n',embedding_output)
        all_encoder_layers = self.encoder(embedding_output, extended_attention_mask)
        sequence_output = all_encoder_layers[-1]
        # print('sequence_output:\n',sequence_output)
        pooled_output = self.pooler(sequence_output)
        return all_encoder_layers, pooled_output

class BertForSequenceClassification(nn.Module):
    """BERT model for classification.
    This module is composed of the BERT model with a linear layer on top of
    the pooled output.

    Example usage:
    ```python
    # Already been converted into WordPiece token ids
    input_ids = torch.LongTensor([[31, 51, 99], [15, 5, 0]])
    input_mask = torch.LongTensor([[1, 1, 1], [1, 1, 0]])
    token_type_ids = torch.LongTensor([[0, 0, 1], [0, 2, 0]])

    config = BertConfig(vocab_size=32000, hidden_size=512,
        num_hidden_layers=8, num_attention_heads=6, intermediate_size=1024)

    num_labels = 2

    model = BertForSequenceClassification(config, num_labels)
    logits = model(input_ids, token_type_ids, input_mask)
    ```
    """
    def __init__(self, config, num_labels):
        super(BertForSequenceClassification, self).__init__()
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, num_labels)

        def init_weights(module):
            if isinstance(module, (nn.Linear, nn.Embedding)):
                # Slightly different from the TF version which uses truncated_normal for initialization
                # cf https://github.com/pytorch/pytorch/pull/5617
                module.weight.data.normal_(mean=0.0, std=config.initializer_range)
            elif isinstance(module, BERTLayerNorm):
                module.beta.data.normal_(mean=0.0, std=config.initializer_range)
                module.gamma.data.normal_(mean=0.0, std=config.initializer_range)
            if isinstance(module, nn.Linear):
                module.bias.data.zero_()
        self.apply(init_weights)

    def forward(self, input_ids, token_type_ids, attention_mask, labels=None):
        _, pooled_output = self.bert(input_ids, token_type_ids, attention_mask)
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits, labels)
            return loss, logits
        else:
            return logits

class BertForQuestionAnswering(nn.Module):
    """BERT model for Question Answering (span extraction).
    This module is composed of the BERT model with a linear layer on top of
    the sequence output that computes start_logits and end_logits

    Example usage:
    ```python
    # Already been converted into WordPiece token ids
    input_ids = torch.LongTensor([[31, 51, 99], [15, 5, 0]])
    input_mask = torch.LongTensor([[1, 1, 1], [1, 1, 0]])
    token_type_ids = torch.LongTensor([[0, 0, 1], [0, 2, 0]])

    config = BertConfig(vocab_size=32000, hidden_size=512,
        num_hidden_layers=8, num_attention_heads=6, intermediate_size=1024)

    model = BertForQuestionAnswering(config)
    start_logits, end_logits = model(input_ids, token_type_ids, input_mask)
    ```
    """
    def __init__(self, config):
        super(BertForQuestionAnswering, self).__init__()
        self.bert = BertModel(config)
        # TODO check with Google if it's normal there is no dropout on the token classifier of SQuAD in the TF version
        # self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.qa_outputs = nn.Linear(config.hidden_size, 2)

        def init_weights(module):
            if isinstance(module, (nn.Linear, nn.Embedding)):
                # Slightly different from the TF version which uses truncated_normal for initialization
                # cf https://github.com/pytorch/pytorch/pull/5617
                module.weight.data.normal_(mean=0.0, std=config.initializer_range)
            elif isinstance(module, BERTLayerNorm):
                module.beta.data.normal_(mean=0.0, std=config.initializer_range)
                module.gamma.data.normal_(mean=0.0, std=config.initializer_range)
            if isinstance(module, nn.Linear):
                module.bias.data.zero_()
        self.apply(init_weights)

    def forward(self, input_ids, token_type_ids, attention_mask, start_positions=None, end_positions=None):
        all_encoder_layers, _ = self.bert(input_ids, token_type_ids, attention_mask)
        sequence_output = all_encoder_layers[-1]
        logits = self.qa_outputs(sequence_output)
        start_logits, end_logits = logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1)
        end_logits = end_logits.squeeze(-1)

        if start_positions is not None and end_positions is not None:
            # If we are on multi-GPU, split add a dimension
            if len(start_positions.size()) > 1:
                start_positions = start_positions.squeeze(-1)
            if len(end_positions.size()) > 1:
                end_positions = end_positions.squeeze(-1)
            # sometimes the start/end positions are outside our model inputs, we ignore these terms
            ignored_index = start_logits.size(1)
            start_positions.clamp_(0, ignored_index)
            end_positions.clamp_(0, ignored_index)

            loss_fct = CrossEntropyLoss(ignore_index=ignored_index)
            start_loss = loss_fct(start_logits, start_positions)
            end_loss = loss_fct(end_logits, end_positions)
            total_loss = (start_loss + end_loss) / 2
            return total_loss
        else:
            return start_logits, end_logits

class BertWithMultiPointer(nn.Module):
    """BERT model for Question Answering (text generation).
    This module is composed of the BERT model with a multi pointer network on top of
    the sequence output to generate the answer.

    Example usage:
    ```python
    # Already been converted into WordPiece token ids
    input_ids = torch.LongTensor([[31, 51, 99], [15, 5, 0]])
    input_mask = torch.LongTensor([[1, 1, 1], [1, 1, 0]])
    token_type_ids = torch.LongTensor([[0, 0, 1], [0, 2, 0]])

    config = BertConfig(vocab_size=32000, hidden_size=512,
        num_hidden_layers=8, num_attention_heads=6, intermediate_size=1024)

    model = BertWithMultiPointer(config)
    loss, probs = model(input_ids, token_type_ids, input_mask)
    ```
    """
    def __init__(self, config):
        super(BertWithMultiPointer, self).__init__()
        self.reduce_dim = config.reduce_dim
        self.decoder_layer = config.decoder_layer
        self.share_decoder_embedd = config.share_embedd
        self.bert = BertModel(config)
        # TODO check with Google if it's normal there is no dropout on the token classifier of SQuAD in the TF version
        # self.dropout = nn.Dropout(config.hidden_dropout_prob)
        # self.qa_outputs = nn.Linear(config.hidden_size, 2)
        self.decoderEmbedding = BERTEmbeddings(config)

        def init_weights(module):
            if isinstance(module, (nn.Linear, nn.Embedding)):
                # Slightly different from the TF version which uses truncated_normal for initialization
                # cf https://github.com/pytorch/pytorch/pull/5617
                module.weight.data.normal_(mean=0.0, std=config.initializer_range)
            elif isinstance(module, BERTLayerNorm):
                module.beta.data.normal_(mean=0.0, std=config.initializer_range)
                module.gamma.data.normal_(mean=0.0, std=config.initializer_range)
            if isinstance(module, nn.Linear) and module.bias is not None:  # XD
                module.bias.data.zero_()
        
        decoder_dim = config.hidden_size
        if self.reduce_dim > 0:
            decoder_dim = self.reduce_dim
            self.linear_answer = nn.Linear(config.hidden_size, decoder_dim, bias=False)  # XD
            self.linear_context = nn.Linear(config.hidden_size, decoder_dim, bias=False)  # XD
            self.ln_answer = LayerNorm(decoder_dim)
            self.ln_context = LayerNorm(decoder_dim)

        self.self_attentive_decoder = TransformerDecoder(decoder_dim, int(decoder_dim/64), decoder_dim*4, self.decoder_layer, 0.2)
        self.dual_ptr_rnn_decoder = DualPtrRNNDecoder(decoder_dim, decoder_dim, dropout=0.2)
        self.vocab_size = config.vocab_size
        self.out = nn.Linear(decoder_dim, self.vocab_size, bias=False)  # XD
        if self.share_decoder_embedd:
            self.project = nn.Linear(decoder_dim, config.hidden_size, bias=False)  # XD
            self.out.weight = self.decoderEmbedding.word_embeddings.weight

        # Paragraph Selector Decoder
        # self.ps_dropout = nn.Dropout(config.hidden_dropout_prob)
        # self.ps_classifier = nn.Linear(config.hidden_size, 1)
        self.apply(init_weights)

    def forward(self, input_ids, token_type_ids, attention_mask, start_positions=None, end_positions=None, answer_ids=None, answer_mask=None):
        
        self.reshape = False
        if input_ids.dim() == 3:
            self.ori_size = input_ids.size()
            new_size = (-1, self.ori_size[-1])
            input_ids = input_ids.view(new_size)
            token_type_ids = token_type_ids.view(new_size)
            attention_mask = attention_mask.view(new_size)
            self.reshape = True

        all_encoder_layers, pooled_output = self.bert(input_ids, token_type_ids, attention_mask)
        sequence_output = all_encoder_layers[-1]
        sequence_output = self.ln_context(self.linear_context(sequence_output)) if self.reduce_dim>0 else sequence_output
        
        context_padding = attention_mask.data == 0
        answer_padding = answer_mask == 0
        self.dual_ptr_rnn_decoder.applyMasks(context_padding)
        
        if self.training:
            answer_embedding = self.decoderEmbedding(answer_ids)
            answer_embedding = self.ln_answer(self.linear_answer(answer_embedding)) if self.reduce_dim>0 else answer_embedding
            self_attended_decoded = self.self_attentive_decoder(answer_embedding[:, :-1].contiguous(), sequence_output, context_padding=context_padding, answer_padding=answer_padding[:, :-1], positional_encodings=False)

            decoder_outputs = self.dual_ptr_rnn_decoder(self_attended_decoded, sequence_output)
            context_question_outputs, context_question_attention, context_question_alignments, vocab_pointer_switches, hidden = decoder_outputs

            probs = self.probs(self.out, context_question_outputs, vocab_pointer_switches, context_question_attention, input_ids)
            probs, gen_targets = mask(answer_ids[:, 1:].contiguous(), probs.contiguous(), pad_idx=0)

            # paragraph selector score
            # pooled_output = self.ps_dropout(pooled_output)
            # ps_score = self.ps_classifier(pooled_output)
            # ps_score = ps_score.view(self.ori_size[0], self.ori_size[1])
            
            loss = F.nll_loss(probs.log(), gen_targets) + F.kl_div(F.log_softmax(ps_score, 1), ps_targets)

            # probs_mask = (answer_ids[:, 1:]==0).unsqueeze(-1).expand_as(probs).contiguous()
            # probs[probs_mask] = 0
            # probs = probs.contiguous().view(-1, probs.size(-1))
            # targets = answer_ids[:, 1:].contiguous().view(-1)
            return loss, None
        else:
            return None, self.greedy(sequence_output, input_ids, answer_ids = answer_ids)

    def greedy(self, self_attended_context, context_ids, oov_to_limited_idx=None, rnn_state=None, answer_ids=None):
        B, TC, C = self_attended_context.size()
        T = 100 # suppose the max length of answer is 15
        outs = self_attended_context.new_full((B, T), 0, dtype=torch.long) # The index of [PAD] is 0. 
        hiddens = [self_attended_context.new_zeros((B, T, C)) for l in range(len(self.self_attentive_decoder.layers) + 1)]
        hiddens[0] = hiddens[0] #+ positional_encodings_like(hiddens[0])
        eos_yet = self_attended_context.new_zeros((B, )).byte()
        answer_scores = self_attended_context.new_zeros((B, ))
        for t in range(T):
            if t == 0:
                embedding = self.decoderEmbedding(self_attended_context.new_full((B, 1), 99, dtype=torch.long), position=t) # the index of "[CLS]" is 101 
            else:
                embedding = self.decoderEmbedding(outs[:, t - 1].unsqueeze(1), position=t)
            embedding = self.ln_answer(self.linear_answer(embedding)) if self.reduce_dim>0 else embedding
            
            hiddens[0][:, t] = hiddens[0][:, t] + embedding.squeeze(1)
            for l in range(len(self.self_attentive_decoder.layers)):
                hiddens[l + 1][:, t] = self.self_attentive_decoder.layers[l].feedforward(
                    self.self_attentive_decoder.layers[l].attention(
                    self.self_attentive_decoder.layers[l].selfattn(hiddens[l][:, t], hiddens[l][:, :t+1], hiddens[l][:, :t+1])
                  , self_attended_context, self_attended_context))
            
            decoder_outputs = self.dual_ptr_rnn_decoder(hiddens[-1][:, t].unsqueeze(1), self_attended_context)
            context_question_outputs, context_question_attention, context_question_alignments, vocab_pointer_switches, hidden = decoder_outputs
            
            probs = self.probs(self.out, context_question_outputs, vocab_pointer_switches, context_question_attention, context_ids)
            
            pred_probs, preds = probs.max(-1)
            preds = preds.squeeze(1)
            pred_probs = pred_probs.log().squeeze(1)
            eos_yet = eos_yet | (preds == 102)  # the index of "[SEP]" is 102
            answer_scores += pred_probs * (1-eos_yet.float())
            outs[:, t] = preds.cpu()
            if eos_yet.all():
                break
        if self.reshape:
            return outs.view((self.ori_size[0], self.ori_size[1], -1)), answer_scores.view((self.ori_size[0], self.ori_size[1], -1))
        return outs, answer_scores

    def probs(self, generator, outputs, vocab_pointer_switches, context_question_attention, context_question_indices, oov_to_limited_idx=None):

        size = list(outputs.size())

        size[-1] = self.vocab_size
        if self.share_decoder_embedd:
            outputs = self.project(outputs)
        scores = generator(outputs.view(-1, outputs.size(-1))).view(size)
        p_vocab = F.softmax(scores, dim=scores.dim()-1)
        scaled_p_vocab = vocab_pointer_switches.expand_as(p_vocab) * p_vocab

        effective_vocab_size = self.vocab_size # + len(oov_to_limited_idx) TODO: add the oov
        if self.vocab_size < effective_vocab_size:
            size[-1] = effective_vocab_size - self.generative_vocab_size
            buff = scaled_p_vocab.new_full(size, EPSILON)
            scaled_p_vocab = torch.cat([scaled_p_vocab, buff], dim=buff.dim()-1)

        # p_context_ptr
        scaled_p_vocab.scatter_add_(scaled_p_vocab.dim()-1, context_question_indices.unsqueeze(1).expand_as(context_question_attention), 
            (1 - vocab_pointer_switches).expand_as(context_question_attention) * context_question_attention)

        return scaled_p_vocab

class DualPtrRNNDecoder(nn.Module):

    def __init__(self, d_in, d_hid, dropout=0.0, num_layers=1):
        super().__init__()
        self.d_hid = d_hid
        self.d_in = d_in
        self.num_layers = num_layers
        self.dropout = nn.Dropout(dropout)

        self.context_question_attn = LSTMDecoderAttention(d_hid, dot=True)  # XD: tried dot=False?

        # self.vocab_pointer_switch = nn.Sequential(Feedforward(self.d_hid + d_in, 1), nn.Sigmoid())
        self.vocab_pointer_switch = nn.Sequential(nn.Dropout(dropout), nn.Linear(self.d_hid + d_in, 1), nn.Sigmoid())  # XD
        
    # XD: to slow, need to be optimized to be fully parallel    
    def forward_slow(self, input, context_question, output=None, hidden=None, context_question_alignment=None):
        context_question_alignment = context_question_alignment if context_question_alignment is not None else self.make_init_output(context_question)

        context_question_outputs, vocab_pointer_switches, context_question_attentions, context_question_alignments = [], [], [], []
        for emb_t in input.split(1, dim=1):
            emb_t = emb_t.squeeze(1)
            context_question_output, context_question_attention, context_question_alignment = self.context_question_attn(emb_t, context_question)            
            vocab_pointer_switch = self.vocab_pointer_switch(torch.cat([emb_t, context_question_output], -1))
            context_question_output = self.dropout(context_question_output)
            context_question_outputs.append(context_question_output)
            vocab_pointer_switches.append(vocab_pointer_switch)
            context_question_attentions.append(context_question_attention)
            context_question_alignments.append(context_question_alignment)
    
        context_question_outputs, vocab_pointer_switches, context_question_attentions = [self.package_outputs(x) for x in [context_question_outputs, vocab_pointer_switches, context_question_attentions]]
        
        return context_question_outputs, context_question_attentions, context_question_alignments, vocab_pointer_switches, hidden


    # def applyMasks(self, context_mask, question_mask):
    def applyMasks(self, context_question_mask):
        self.context_question_attn.applyMasks(context_question_mask)

    def make_init_output(self, context):
        batch_size = context.size(0)
        h_size = (batch_size, self.d_hid)
        return context.new_zeros(h_size)

    def package_outputs(self, outputs):
        outputs = torch.stack(outputs, dim=1)
        return outputs

