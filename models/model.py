from .gcn import *
from .cls_cvt import *
from torch.nn import Parameter
from .agcn import *
import pickle


class LAGNet(nn.Module):
    def __init__(self, config, num_classes, adj_file=None):
        super(LAGNet, self).__init__()
        msvit_spec = config.MODEL.SPEC
        self.features = ConvolutionalVisionTransformer(
            in_chans=3,
            num_classes=num_classes,
            act_layer=QuickGELU,
            norm_layer=partial(LayerNorm, eps=1e-5),
            init=getattr(msvit_spec, 'INIT', 'trunc_norm'),
            spec=msvit_spec
        )
        self.sig = nn.Sigmoid()
        
        
        fp = open('dataset/vec.pickle', 'rb+')
        feat = pickle.load(fp).cuda(non_blocking=True)
        self.agcn = AttentionGCN(num_classes, feat, adj_file, layers=2)

        self.num_classes = num_classes
        self.relu = nn.LeakyReLU(0.5)
        self.drop = nn.Dropout(0.3)
        
        #label fusion
        self.laba = nn.MultiheadAttention(num_classes, 7, dropout=0.2, batch_first=True)
        
        self.fc1 = nn.Linear(num_classes, 1024)
        self.classifier = nn.Linear(1024, num_classes)
    def forward(self, feature):
        feature = self.features(feature)
        #feature = self.pooling(feature)
        #feature = feature.view(feature.size(0), -1)

        feat = self.sig(feature)
        x = self.agcn(feat)
        
        x, _ = self.laba(feature, x, feature)

        x = self.fc1(x)
        x = self.drop(x)
        x = self.relu(x)
        x = self.classifier(x)
        return x

    def get_config_optim(self, lr, lrp):
        return [
                {'params': self.features.parameters(), 'lr': lr * lrp},
                {'params': self.gc1.parameters(), 'lr': lr},
                {'params': self.gc2.parameters(), 'lr': lr},
                ]



#单独Cvt
def getCvt(config):
    msvit_spec = config.MODEL.SPEC
    msvit = ConvolutionalVisionTransformer(
        in_chans=3,
        num_classes=config.MODEL.NUM_CLASSES,
        act_layer=QuickGELU,
        norm_layer=partial(LayerNorm, eps=1e-5),
        init=getattr(msvit_spec, 'INIT', 'trunc_norm'),
        spec=msvit_spec
    )

    # if config.MODEL.INIT_WEIGHTS:
    #     msvit.init_weights(
    #         config.MODEL.PRETRAINED,
    #         config.MODEL.PRETRAINED_LAYERS,
    #         config.VERBOSE
    #     )

    return msvit


def getLAGNet(config):
    model = LAGNet(config=config, num_classes=config.MODEL.NUM_CLASSES, adj_file=config.DATA.ADJ_PATH)
    return model