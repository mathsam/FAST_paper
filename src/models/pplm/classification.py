import torch 

def classification_head(inputs_embeds, features, model, mode): 
    """
    Inherit the model embedding and follow with classification head
    """
    #x = features[:, 0, :]  # take <s> token (equiv. to [CLS])
    classifier = model.classifier
    encoder = model.roberta
    if mode == "use_input_embeds": 
        outputs = encoder(inputs_embeds=inputs_embeds)
        sequence_output = outputs[0]
        logits = classifier(sequence_output)
    elif mode == "use_hidden_state": 
        x = classifier.dropout(features)
        x = classifier.dense(x)
        x = torch.tanh(x)
        x = classifier.dropout(x)
        logits = classifier.out_proj(x)
    return logits

