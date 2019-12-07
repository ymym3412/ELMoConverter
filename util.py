import torch
import numpy as np
import h5py


def convert_config(config):
    """
    convert ELMoForManyLangs config to AllenNLP
    """
    allennlp_config = {}

    char_cnn_dict = {}
    char_cnn_dict['activation'] = config['token_embedder']['activation']
    char_cnn_dict['filters'] = config['token_embedder']['filters']
    char_cnn_dict['n_highway'] = config['token_embedder']['n_highway']
    char_cnn_dict['embedding'] = {'dim': config['token_embedder']['char_dim']}
    char_cnn_dict['max_characters_per_token'] = config['token_embedder']['max_characters_per_token']
    allennlp_config['char_cnn'] = char_cnn_dict

    lstm_dict = {}
    # Currently, AllenNLP support lstm with skip connection only
    lstm_dict['use_skip_connections'] = True
    lstm_dict['projection_dim'] = config['encoder']['projection_dim']
    lstm_dict['cell_clip'] = config['encoder']['cell_clip']
    lstm_dict['proj_clip'] = config['encoder']['proj_clip']
    lstm_dict['dim'] = config['encoder']['dim']
    lstm_dict['n_layers'] = config['encoder']['n_layers']
    allennlp_config['lstm'] = lstm_dict

    return allennlp_config


def create_lstm_weight(hdf5_file, encoder):
    state_dict = encoder.state_dict()
    directions = ['forward', 'backward']
    layers = [0, 1]
    for direction in directions:
        for layer in layers:
            direction_num = 0 if direction == 'forward' else 1
            base_key = f'{direction}_layer_{layer}.'
            concat_weight = torch.cat([state_dict[base_key + 'input_linearity.weight'], state_dict[base_key + 'state_linearity.weight']], dim=1)
            # weight
            hdf5_file.create_dataset(
                f'RNN_{direction_num}/RNN/MultiRNNCell/Cell{layer}/LSTMCell/W_0',
                data=np.transpose(concat_weight.cpu())
            )
            # bias
            hdf5_file.create_dataset(
                f'RNN_{direction_num}/RNN/MultiRNNCell/Cell{layer}/LSTMCell/B',
                data=np.transpose(state_dict[base_key + 'state_linearity.bias'].cpu())
            )
            # projection
            hdf5_file.create_dataset(
                f'RNN_{direction_num}/RNN/MultiRNNCell/Cell{layer}/LSTMCell/W_P_0',
                data=np.transpose(state_dict[base_key + 'state_projection.weight'].cpu())
            )


def create_char_embed_weight(hdf5_file, embedding_layer):
    hdf5_file.create_dataset('char_embed', data=embedding_layer.state_dict()['embedding.weight'].cpu().numpy())


def create_CNN_weight(hdf5_file, convolutions):
    for i, conv1d in enumerate(convolutions):
        state_dict = conv1d.state_dict()
        weight = state_dict['weight'].cpu().numpy()  # width * char_emb_dim * out_ch
        bias = state_dict['bias'].cpu().numpy()

        weight = np.transpose(weight)
        weight = weight.reshape(1, *weight.shape)  # 1 * iut_ch * char_emb_dim * width
        hdf5_file.create_dataset('CNN/W_cnn_{}'.format(i), data=weight)
        hdf5_file.create_dataset('CNN/b_cnn_{}'.format(i), data=bias)


def create_hightway_weight(hdf5_file, hightway_layers):
    """
    In ELMoForManyLangs, highway layer has linear layer.
    The weight of linear layer consist of two part, carry weight and non-linear weight.
    First half of weight is carry weight, and latter half is non-linear part.
    See also https://medium.com/jim-fleming/highway-networks-with-tensorflow-1e6dfa667daa
    """
    for i, layer in enumerate(hightway_layers):
        state_dict = layer.state_dict()
        # input_dim * 2, input_dim
        weight = state_dict['weight'].cpu().numpy()
        # input_dim
        bias = state_dict['bias'].cpu().numpy()

        input_dim = weight.shape[1]
        w_carry = weight[:input_dim, :]
        w_transform = weight[input_dim:, :]
        b_carry = bias[:input_dim]
        b_transform = bias[input_dim:]

        hdf5_file.create_dataset('CNN_high_{}/W_carry'.format(i), data=np.transpose(w_carry))
        hdf5_file.create_dataset('CNN_high_{}/W_transform'.format(i), data=np.transpose(w_transform))
        hdf5_file.create_dataset('CNN_high_{}/b_carry'.format(i), data=b_carry)
        hdf5_file.create_dataset('CNN_high_{}/b_transform'.format(i), data=b_transform)


def create_projection_weight(hdf5_file, projection, word_dim):
    # In ELMoForManyLangs, embedding is created by concat of word emb and char emb.
    # So transger only char emb projection.
    weight = projection.state_dict()['weight'].cpu().numpy()[:, word_dim:]
    bias = projection.state_dict()['bias'].cpu().numpy()
    hdf5_file.create_dataset('CNN_proj/W_proj', data=np.transpose(weight))
    hdf5_file.create_dataset('CNN_proj/b_proj', data=bias)


def create_elmo_h5_from_embedder(embedder, h5_path, config):
    with h5py.File(h5_path, 'w') as f:
        create_lstm_weight(f, embedder.model.encoder)
        create_char_embed_weight(f, embedder.model.token_embedder.char_emb_layer)
        create_CNN_weight(f, embedder.model.token_embedder.convolutions)
        create_hightway_weight(f, embedder.model.token_embedder.highways._layers)
        create_projection_weight(f, embedder.model.token_embedder.projection, word_dim=config['token_embedder']['word_dim'])
