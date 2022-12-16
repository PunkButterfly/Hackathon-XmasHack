import torch.nn as nn
from transformers.modeling_outputs import SequenceClassifierOutput
from transformers import AutoModel, AutoConfig
from torch.optim import AdamW


class BertForSequenceClassification(nn.Module):
    def __init__(
        self, 
        pretrained_model_name : str,
        num_labels : int,
        dropout : float = 0.25
    ):
        super().__init__()
        
        # Form the model configuration from passed pretrained_model_name and num_labels
        config = AutoConfig.from_pretrained(
            pretrained_model_name, 
            num_labels=num_labels
        )
       
        self.model = AutoModel.from_pretrained(
            pretrained_model_name, 
            config=config
        )
        self.num_labels = num_labels
        self.pre_classifier = nn.Linear(config.hidden_size, config.hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)  
    
    def forward(self, input_ids, attention_mask=None, head_mask=None, labels=None, return_dict=False):
        """
        Compute class probabilities for the input sequence
            Args:
                features - torch DataLoader containing 
        """
        
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.model(
            input_ids=input_ids, 
            attention_mask=attention_mask, 
            head_mask=head_mask,
            return_dict=return_dict,
        )

        hidden_state = outputs[0]  # (bs, seq_len, dim)
        # For each sample in batch we only need an output embedding for [CLS] token
        pooled_output = hidden_state[:, 0]  # (bs, dim)
        pooled_output = self.pre_classifier(pooled_output)  # (bs, dim)
        pooled_output = nn.ReLU()(pooled_output)  # (bs, dim)
        pooled_output = self.dropout(pooled_output)  # (bs, dim)
        logits = self.classifier(pooled_output)  # (bs, num_labels)

        loss = None
        if labels is not None:
            if self.num_labels == 1:
                #  We are doing regression
                loss_fct = nn.MSELoss()
                loss = loss_fct(logits.view(-1), labels.view(-1))
            else:
                loss_fct = nn.CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output
        
        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )