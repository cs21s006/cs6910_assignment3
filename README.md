# CS6910 Assignment 3

[Link to Weights & Biases Report](https://wandb.ai/cs21s006_cs21s043/Assigment3_Q2/reports/Assignment-3-Report--VmlldzoxOTYwNjM2)


## Setup

**Note:** It is recommended to create a new python virtual environment before installing dependencies.

```
pip install requirements.txt
python train.py
```

## Train and Evaluate

The code is flexible such that the dimension of the input character embeddings, the hidden states of the encoders and decoders, the cell (RNN, LSTM, GRU) and the number of layers in the encoder and decoder can be changed using command line arguments.

To train the model, the training script can be invoked as follows

```
python train.py --input_embedding_size 256 --hidden_units 128 --cell_type LSTM --num_encoder_layers 2 --num_decoder_layers 4
```

### Arguments

| Name | Default Value | Description |
| :---: | :-------------: | :----------- |
| `-e`, `--epochs` | 5 |  Number of epochs to trainmodel.|
| `-b`, `--batch_size` | 16 | Batch size used to train to optimize model parameters |
| `-lr`, `--learning_rate` | 0.0003 | Learning rate used 
| `-c`, `--cell_type` | GRU | choices:  ['RNN', 'LSTM', 'GRU'] | 
| `-d`, `--dropout` | 0.3 | dropout value | 
| `-hs`, `--hidden_units` | 128 | Number of hidden units. | 
| `-es`, `--input_embedding_size` | 128 | Embedding dimensions. |
| `-el`, `--num_encoder_layers` | 1 | Number of encoder layers |
| `-dl`, `--num_decoder_layers` | 1 | Number of decoder layers |
| `-beam`, `--beam_sizes` | 1 | Beam size used for beam search |
| `-dec_str`, `--decoding_strategy` | greedy | Decoding strategy used to decode inputs. |

## Quick Links

* [Question 1](train.py)
* [Question 2](Question_2.ipynb)
* [Question 4](Question_4.ipynb)
* [Question 5(a)](Question%205.ipynb)
* [Question 5(b,c,d)](Question%205%20Test%20Visualize%20Attention%20HeatMaps.ipynb)
* [Question 6](Question_6.ipynb)
* [Question 7](https://github.com/cs21s006/cs6910_assignment3/)
* [Question 8](Question%208.ipynb)

## Team 
* [CS21S043](https://github.com/jainsaurabh426)
* [CS21S006](https://github.com/cs21s006)