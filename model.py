import torch
import torch.nn as nn

class KuaishouDIN(nn.Module):

    def __init__(self, phase, user_net, photo_fc_net,text_net, cover_net, photo_net, head_net, photo_count, num_classes ):
        super(KuaishouDIN, self).__init__()
        self.phase = phase
        self.user_size = 1
        self.photo_feature_size = 2048
        self.text_size = 5
        self.face_size = 4
        self.num_classes = num_classes

        self.photo_count = photo_count

        # KuaishouDIN network
        self.user_net = nn.ModuleList(user_net)

        self.photo_fc_net = nn.ModuleList(photo_fc_net)
        self.text_net = nn.ModuleList(text_net)
        self.cover_net = nn.ModuleList(cover_net)

        for i in range(photo_count):
            self.photos_net[i] = nn.ModuleList(photo_net)

        self.head_net = nn.ModuleList(head_net( (photo_count+2)*32 ))

        if phase == 'test':
            self.softmax = nn.Softmax(dim=-1)


    def forward(self, train_data, data_feature):
        user_id = train_data[:, 0]
        hist = train_data[:, 1]
        candidate = train_data[:, 2]

        # user net
        user_emb = user_net[0](user_id)
        user_vec = user_emb.view(-1)
        user_vec = user_net[1](user_vec)




def user_net(input_size = 1):
    layers =[]
    layers = nn.Embedding(input_size, 5)
    layers += nn.Embedding(input_size*5, 32)
    return layers

# photo feature net
def photo_fc_net(input_size = 2048):
    layers = []
    layers += nn.Linear(input_size, 512)
    layers += nn.Linear(512, 128)
    layers += nn.Linear(128, 32)
    return layers

# text content net
def text_net(num_keywords= 5):
    layers = []
    layers += nn.Embedding(num_keywords, 5) # data between emb and linear show change shape
    layers += nn.Linear(num_keywords*5, 64)
    layers += nn.Linear(64, 32)
    return layers

def cover_net(input_size = 4):
    layers =[]
    layers += nn.Linear(input_size, 64)
    layers += nn.Linear(64, 32)
    return layers

def photo_net(input_size = 128):
    layers = []
    layers += nn.Linear(input_size, 64)
    layers += nn.Linear(64, 32)
    return layers

def head_net(input_size = 0):
    layers = []
    layers += nn.Linear(input_size, 1024)
    layers += nn.Linear(1024, 128)
    layers += nn.Linear(128, 32)
    layers += nn.Linear(32, 1)
    return layers


def build_KuaishouDIN():
    print(1)










