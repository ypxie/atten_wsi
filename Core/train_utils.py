import torch
from torch.autograd import Variable

def load_partial_state_dict(model, state_dict):
    
    own_state = model.state_dict()
    #print('own_Dict', own_state.keys(), 'state_Dict',state_dict.keys())
    for a,b in zip( own_state.keys(), state_dict.keys()):
        print(a,'_from model =====_loaded: ', b)
    for name, param in state_dict.items():
        if name is "device_id":
            pass
        else:
            if name not in own_state:
                print('unexpected key "{}" in state_dict'.format(name))
            #if isinstance(param, Parameter):
                # backwards compatibility for serialized parameters
            param = param.data
            try:
                own_state[name].copy_(param)
            except:
                print('While copying the parameter named {}, whose dimensions in the model are'
                    ' {} and whose dimensions in the checkpoint are {}, ...'.format(
                        name, own_state[name].size(), param.size()))
                #raise
    print ('>> load partial state dict: {} initialized'.format(len(state_dict)))


def adding_grad_noise(model, eta, time_step):
    for p in model.parameters(): 
        sigma = eta/time_step**0.55
        this_grad = p.grad
        
        noise = sigma*Variable( torch.randn_like(this_grad)  )
        this_grad += noise

