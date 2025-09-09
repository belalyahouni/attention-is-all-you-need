import torch
# create mask for all src tokens <pad>
def create_encoder_mask(input_ids, pad_token_id):
    encoder_mask = (input_ids != pad_token_id).unsqueeze(0).unsqueeze(0).int()
    return encoder_mask

# pad_token_id = 1

# create casual mask for future tokens and pad tokens
def create_decoder_mask(input_ids, pad_token_id):
    pad_mask = (input_ids != pad_token_id).unsqueeze(0).int()
    casual_mask = torch.triu(torch.ones((1, input_ids.size(0), input_ids.size(0))), diagonal=1).type(torch.int)
    casual_mask = casual_mask == 0

    decoder_mask = pad_mask & casual_mask
    return decoder_mask