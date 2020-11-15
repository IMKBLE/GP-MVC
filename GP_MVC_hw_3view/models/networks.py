import torch
import torch.nn as nn
from torch.autograd import Variable
from pdb import set_trace as st
import torch.nn.functional as F
import torch.optim as optim

###########################
# Autoencoder
###########################
# You can modify the model using convolutional layer
class AutoEncoder(nn.Module):
    def __init__(self, input_A, input_B, input_C):
        super(AutoEncoder, self).__init__()
        # input_A = 1750 input_B = 79
        self.layer1_1 = nn.Linear(input_A, 200, bias = False)
        self.layer1_2 = nn.Linear(input_B, 200, bias = False)
        self.layer1_3 = nn.Linear(input_C, 200, bias = False)
        self.layer2 = nn.Linear(200, 150)
        self.layer3 = nn.Linear(150, 100)
        self.layer4 = nn.Linear(100, 150)
        self.layer5 = nn.Linear(150, 200)
#        self.layer5_2 = nn.Linear(300, 79)
        self.layer6_1 = nn.Linear(200, input_A)
        self.layer6_2 = nn.Linear(200, input_B)
        self.layer6_3 = nn.Linear(200, input_C)
        self.drop = 0.5
        self.beta1 = torch.nn.Parameter(torch.Tensor([0.5]))
        self.beta2 = torch.nn.Parameter(torch.Tensor([0.5]))

    def forward(self, x1, x2, x3):
        x1 = F.dropout(F.relu(self.layer1_1(x1.view(-1, 216))), self.drop)
        x2 = F.dropout(F.relu(self.layer1_2(x2.view(-1, 76))), self.drop)
        x3 = F.dropout(F.relu(self.layer1_3(x3.view(-1, 64))), self.drop)

        x = F.dropout(F.relu(self.layer2(torch.cat((x1, x2, x3), 1))), self.drop)
#        x = F.dropout(F.relu(self.layer3(x)), self.drop)
#        x = F.dropout(F.relu(self.layer4(x)), self.drop)

        out1 = F.relu(self.layer5_1(x))
#        out1 = F.sigmoid(self.layer6_1(out1))
        out1 = F.tanh(self.layer6_1(out1))
        out2 = F.relu(self.layer5_2(x))
#        out2 = F.sigmoid(self.layer6_2(out2))
        out2 = F.tanh(self.layer6_2(out2))
        out3 = F.tanh(self.layer6_3(out2))
        
        out1 = out1.view(1,1,216)
        out2 = out2.view(1,1,76)
        out3 = out3.view(1,1,64)
        return out1, out2, out3
    

    def forward_ac2b(self, x1,x3):
        x1 = F.dropout(F.relu(self.layer1_1(x1.view(-1, 216))), self.drop)
        x3 = F.dropout(F.relu(self.layer1_3(x3.view(-1, 64))), self.drop)
        
        x =  (x1+x3)/2      
        x = F.dropout(F.relu(self.layer2(x)), self.drop)   
        self.com1  = F.dropout(F.relu(self.layer3(x)), self.drop)        
        
#        x = F.dropout(F.relu(self.layer2(x1)), self.drop)
#        self.com = F.dropout(F.relu(self.layer3(x)), self.drop)        
        x = F.dropout(F.relu(self.layer4(self.com1)), self.drop)
        x = F.dropout(F.relu(self.layer5(x)), self.drop)
        
#        out1 = F.relu(self.layer5_1(x))
#        out1 = F.sigmoid(self.layer6_1(x))
        out1 = F.tanh(self.layer6_1(x))
#        out2 = F.relu(self.layer5_2(x))
#        out2 = F.sigmoid(self.layer6_2(x))
        out2 = F.tanh(self.layer6_2(x))
        
        out3 = F.tanh(self.layer6_3(x))
        
        out1 = out1.view(1,1,216)
        out2 = out2.view(1,1,76)
        out3 = out3.view(1,1,64)
        return out1, out2, out3, self.com1
    
    def forward_bc2a(self, x2,x3):
        x2 = F.dropout(F.relu(self.layer1_2(x2.view(-1, 76))), self.drop)
        x3 = F.dropout(F.relu(self.layer1_3(x3.view(-1, 64))), self.drop)
        
        x =  (x2+x3)/2      
        x = F.dropout(F.relu(self.layer2(x)), self.drop)
        
        self.com2  = F.dropout(F.relu(self.layer3(x)), self.drop)
        
#        x = F.dropout(F.relu(self.layer2(x2)), self.drop)
#        self.com = F.dropout(F.relu(self.layer3(x)), self.drop)        
        x = F.dropout(F.relu(self.layer4(self.com2)), self.drop)
        x = F.dropout(F.relu(self.layer5(x)), self.drop)
        
#        out1 = F.relu(self.layer5_1(x))
#        out1 = F.sigmoid(self.layer6_1(x))
        out1 = F.tanh(self.layer6_1(x))
#        out2 = F.relu(self.layer5_2(x))
#        out2 = F.sigmoid(self.layer6_2(x))
        out2 = F.tanh(self.layer6_2(x))
        out3 = F.tanh(self.layer6_3(x))
        
        out1 = out1.view(1,1,216)
        out2 = out2.view(1,1,76)
        out3 = out3.view(1,1,64)
        return out1, out2, out3, self.com2
    
    def forward_ab2c(self, x1,x2):
        x1 = F.dropout(F.relu(self.layer1_1(x1.view(-1, 216))), self.drop)
        x2 = F.dropout(F.relu(self.layer1_2(x2.view(-1, 76))), self.drop)
        
        x =  (x2+x1)/2      
        x = F.dropout(F.relu(self.layer2(x)), self.drop)        
        self.com3  = F.dropout(F.relu(self.layer3(x)), self.drop)
        
#        x = F.dropout(F.relu(self.layer2(x2)), self.drop)
#        self.com = F.dropout(F.relu(self.layer3(x)), self.drop)        
        x = F.dropout(F.relu(self.layer4(self.com3)), self.drop)
        x = F.dropout(F.relu(self.layer5(x)), self.drop)
        
#        out1 = F.relu(self.layer5_1(x))
#        out1 = F.sigmoid(self.layer6_1(x))
        out1 = F.tanh(self.layer6_1(x))
#        out2 = F.relu(self.layer5_2(x))
#        out2 = F.sigmoid(self.layer6_2(x))
        out2 = F.tanh(self.layer6_2(x))
        out3 = F.tanh(self.layer6_3(x))
        
        out1 = out1.view(1,1,216)
        out2 = out2.view(1,1,76)
        out3 = out3.view(1,1,64)
        return out1, out2, out3, self.com3
    
    def forward_commonZ(self, x1, x2, x3):       
        x1 = F.dropout(F.relu(self.layer1_1(x1.view(-1, 216))), self.drop)
        x2 = F.dropout(F.relu(self.layer1_2(x2.view(-1, 76))), self.drop)
        x3 = F.dropout(F.relu(self.layer1_3(x3.view(-1, 64))), self.drop)
        
        x1 = F.dropout(F.relu(self.layer2(x1)), self.drop)        
        x1 = F.dropout(F.relu(self.layer3(x1)), self.drop)
        x2 = F.dropout(F.relu(self.layer2(x2)), self.drop)        
        x2 = F.dropout(F.relu(self.layer3(x2)), self.drop)
        x3 = F.dropout(F.relu(self.layer2(x3)), self.drop)        
        x3 = F.dropout(F.relu(self.layer3(x3)), self.drop)
        #self.com = (x1+x2+x3)/3
        self.com = 0.5*torch.sigmoid(self.beta1)*x1 + 0.5*torch.sigmoid(self.beta2)*x2 + (1- 0.5*torch.sigmoid(self.beta1)-0.5*torch.sigmoid(self.beta2))*x3
		
        return self.com
    
    def forward_2common(self, x1, x2, x3):       
        x1 = F.dropout(F.relu(self.layer1_1(x1.view(-1, 216))), self.drop)
        x2 = F.dropout(F.relu(self.layer1_2(x2.view(-1, 76))), self.drop)
        x3 = F.dropout(F.relu(self.layer1_3(x3.view(-1, 64))), self.drop)
        
        x1 = F.dropout(F.relu(self.layer2(x1)), self.drop)        
        x1 = F.dropout(F.relu(self.layer3(x1)), self.drop)
        x2 = F.dropout(F.relu(self.layer2(x2)), self.drop)        
        x2 = F.dropout(F.relu(self.layer3(x2)), self.drop)
        x3 = F.dropout(F.relu(self.layer2(x3)), self.drop)        
        x3 = F.dropout(F.relu(self.layer3(x3)), self.drop)
        self.com12 = (x1+x2)/2
        self.com13 = (x1+x3)/2
        self.com23 = (x3+x2)/2

        return self.com12, self.com13, self.com23
    
def define_AE(input_nc, output_nc, put_nc, gpu_ids=[]):
    NetAE = None
    use_gpu = len(gpu_ids) > 0
    if use_gpu:
        assert (torch.cuda.is_available())
    NetAE = AutoEncoder(input_nc, output_nc, put_nc)

    if len(gpu_ids) > 0:
        #NetAE.cuda(device_id=gpu_ids[0])
        NetAE.cuda(gpu_ids[0])
    NetAE.apply(weights_init)
    return NetAE


class Clustering(nn.Module):
    def __init__(self, K,d):
        super(Clustering, self).__init__()
        # input_A = 784  input_B = 784
        #self.commonz = input1
        self.weights = nn.Parameter(torch.randn(K,d).cuda(), requires_grad=True)
#        self.layer1 = nn.Linear(d, K, bias = False)

    def forward(self, comz):
#         x1 =self.layer1(comz)
#         return x1
        q1 = 1.0 / (1.0 + (torch.sum(torch.pow(torch.unsqueeze(comz, 1) - self.weights, 2), 2)))
        q = torch.t(torch.t(q1) / torch.sum(q1))
        loss_q = torch.log(q)
        return loss_q, q
    

def define_clustering(K,d, gpu_ids=[]):
    Net_clu = None
    use_gpu = len(gpu_ids) > 0
    if use_gpu:
        assert (torch.cuda.is_available())
    Net_clu = Clustering(K,d)

    if len(gpu_ids) > 0:
        #NetAE.cuda(device_id=gpu_ids[0])
        Net_clu.cuda(gpu_ids[0])
    #Net_clu.apply(weights_init)
    return Net_clu

###############################################################################
# Functions
###############################################################################


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1 or  classname.find('InstanceNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


def define_G(input_nc, output_nc, ngf, which_model_netG, norm, gpu_ids=[]):
    netG = None
    use_gpu = len(gpu_ids) > 0
    if norm == 'batch':
        norm_layer = nn.BatchNorm2d
    elif norm == 'instance':
        norm_layer = InstanceNormalization
    else:
        print('normalization layer [%s] is not found' % norm)
    if use_gpu:
        assert(torch.cuda.is_available())

    if which_model_netG == 'resnet_9blocks':
        netG = ResnetGenerator(input_nc, output_nc, ngf, norm_layer, n_blocks=9, gpu_ids=gpu_ids)
    elif which_model_netG == 'resnet_6blocks':
        netG = ResnetGenerator(input_nc, output_nc, ngf, norm_layer, n_blocks=6, gpu_ids=gpu_ids)
    elif which_model_netG == 'unet_128':
        netG = UnetGenerator(input_nc, output_nc, 7, ngf, norm_layer, gpu_ids=gpu_ids)
    elif which_model_netG == 'unet_256':
        netG = UnetGenerator(input_nc, output_nc, 8, ngf, norm_layer, gpu_ids=gpu_ids)
    else:
        print('Generator model name [%s] is not recognized' % which_model_netG)
    if len(gpu_ids) > 0:
        netG.cuda(gpu_ids[0])
        #netG.cuda(device_id=gpu_ids[0])
    netG.apply(weights_init)
    return netG


def define_D(input_nc, ndf, which_model_netD,
             n_layers_D=3, use_sigmoid=False, gpu_ids=[]):
    netD = None
    use_gpu = len(gpu_ids) > 0
    if use_gpu:
        assert(torch.cuda.is_available())
    if which_model_netD == 'basic':
        netD = define_D(input_nc, ndf, 'n_layers', use_sigmoid=use_sigmoid, gpu_ids=gpu_ids)
    elif which_model_netD == 'n_layers':
        netD = NLayerDiscriminator(input_nc, ndf, n_layers_D, use_sigmoid, gpu_ids=gpu_ids)
    else:
        print('Discriminator model name [%s] is not recognized' %
              which_model_netD)
    if use_gpu:
      #  netD.cuda(device_id=gpu_ids[0])
         netD.cuda(gpu_ids[0])
    netD.apply(weights_init)
    return netD


def print_network(net):
    num_params = 0
    for param in net.parameters():
        num_params += param.numel()
    print(net)
    print('Total number of parameters: %d' % num_params)


##############################################################################
# Classes
##############################################################################


# Defines the GAN loss used in LSGAN.
# It is basically same as MSELoss, but it abstracts away the need to create
# the target label tensor that has the same size as the input
class GANLoss(nn.Module):
    def __init__(self, use_lsgan=True, target_real_label=1.0, target_fake_label=0.0,
                 tensor=torch.FloatTensor):
        super(GANLoss, self).__init__()
        self.real_label = target_real_label
        self.fake_label = target_fake_label
        self.real_label_var = None
        self.fake_label_var = None
        self.Tensor = tensor
        if use_lsgan:
            self.loss = nn.MSELoss()
        else:
            self.loss = nn.BCELoss()

    def get_target_tensor(self, input, target_is_real):
        target_tensor = None
        if target_is_real:
            create_label = ((self.real_label_var is None) or
                            (self.real_label_var.numel() != input.numel()))
            if create_label:
                real_tensor = self.Tensor(input.size()).fill_(self.real_label)
                self.real_label_var = Variable(real_tensor, requires_grad=False)
            target_tensor = self.real_label_var
        else:
            create_label = ((self.fake_label_var is None) or
                            (self.fake_label_var.numel() != input.numel()))
            if create_label:
                fake_tensor = self.Tensor(input.size()).fill_(self.fake_label)
                self.fake_label_var = Variable(fake_tensor, requires_grad=False)
            target_tensor = self.fake_label_var
        return target_tensor

    def __call__(self, input, target_is_real):
        target_tensor = self.get_target_tensor(input, target_is_real)
        return self.loss(input, target_tensor)


# Defines the generator that consists of Resnet blocks between a few
# downsampling/upsampling operations.
# Code and idea originally from Justin Johnson's architecture.
# https://github.com/jcjohnson/fast-neural-style/
class ResnetGenerator(nn.Module):
    def __init__(self, input_nc, output_nc, ngf=64, norm_layer=nn.BatchNorm2d, n_blocks=6, gpu_ids=[]):
        assert(n_blocks >= 0)
        super(ResnetGenerator, self).__init__()
        self.input_nc = input_nc #input_nc
        self.output_nc = output_nc
        self.ngf = ngf
        self.gpu_ids = gpu_ids
        #####################################################
        ####WQQQQQQ#####
        model = [nn.Linear(self.input_nc,64),
                 nn.ReLU(True),
                 nn.Linear(64,self.output_nc),
                 nn.ReLU()]
        #nn.ReLU()   nn.Tanh()
#        if input_nc<100:
#            model = [nn.Linear(self.input_nc,64),
#                 nn.ReLU(True),
#                 nn.Linear(64,self.output_nc),
#                 nn.Tanh()]
#        else:
#            model = [nn.Linear(self.input_nc,600),
#                 nn.ReLU(True),
#                 nn.Linear(600,self.output_nc),
#                 nn.Tanh()]
        
#        model = [nn.Conv2d(input_nc, ngf, kernel_size=7, padding=3),
#                 norm_layer(ngf),
#                 nn.ReLU(True)]
#        model = [nn.Conv2d(input_nc, ngf, kernel_size=7, padding=3),
#                 norm_layer(ngf),
#                 nn.ReLU(True)]
#
#        n_downsampling = 2
#        for i in range(n_downsampling):
#            mult = 2**i
#            model += [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3,
#                                stride=2, padding=1),
#                      norm_layer(ngf * mult * 2),
#                      nn.ReLU(True)]
#
#        mult = 2**n_downsampling
#        for i in range(n_blocks):
#            model += [ResnetBlock(ngf * mult, 'zero', norm_layer=norm_layer)]
#
#        for i in range(n_downsampling):
#            mult = 2**(n_downsampling - i)
#            model += [nn.ConvTranspose2d(ngf * mult, int(ngf * mult / 2),
#                                         kernel_size=3, stride=2,
#                                         padding=1, output_padding=1),
#                      norm_layer(int(ngf * mult / 2)),
#                      nn.ReLU(True)]
#
#        model += [nn.Conv2d(ngf, output_nc, kernel_size=7, padding=3)]
       
        #self.model += [nn.Tanh()]

        self.model = nn.Sequential(*model)

    def forward(self, input):
        if isinstance(input.data, torch.cuda.FloatTensor) and self.gpu_ids:
            #return nn.parallel.data_parallel(self.model, input, self.gpu_ids)
            #print(input.view(784))
            return self.model(input.view(self.input_nc))
        else:
            return self.model(input)


# Define a resnet block
class ResnetBlock(nn.Module):
    def __init__(self, dim, padding_type, norm_layer):
        super(ResnetBlock, self).__init__()
        self.conv_block = self.build_conv_block(dim, padding_type, norm_layer)

    def build_conv_block(self, dim, padding_type, norm_layer):
        conv_block = []
        p = 0
        # TODO: support padding types
        assert(padding_type == 'zero')
        p = 1

        # TODO: InstanceNorm
        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p),
                       norm_layer(dim),
                       nn.ReLU(True)]
        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p),
                       norm_layer(dim)]

        return nn.Sequential(*conv_block)

    def forward(self, x):
        out = x + self.conv_block(x)
        return out


# Defines the Unet generator.
# |num_downs|: number of downsamplings in UNet. For example,
# if |num_downs| == 7, image of size 128x128 will become of size 1x1
# at the bottleneck
class UnetGenerator(nn.Module):
    def __init__(self, input_nc, output_nc, num_downs, ngf=64,
                 norm_layer=nn.BatchNorm2d, gpu_ids=[]):
        super(UnetGenerator, self).__init__()
        self.gpu_ids = gpu_ids

        # currently support only input_nc == output_nc
        assert(input_nc == output_nc)

        # construct unet structure
        unet_block = UnetSkipConnectionBlock(ngf * 8, ngf * 8, innermost=True)
        for i in range(num_downs - 5):
            unet_block = UnetSkipConnectionBlock(ngf * 8, ngf * 8, unet_block)
        unet_block = UnetSkipConnectionBlock(ngf * 4, ngf * 8, unet_block)
        unet_block = UnetSkipConnectionBlock(ngf * 2, ngf * 4, unet_block)
        unet_block = UnetSkipConnectionBlock(ngf, ngf * 2, unet_block)
        unet_block = UnetSkipConnectionBlock(output_nc, ngf, unet_block, outermost=True)

        self.model = unet_block

    def forward(self, input):
        if isinstance(input.data, torch.cuda.FloatTensor) and self.gpu_ids:
            return nn.parallel.data_parallel(self.model, input, self.gpu_ids)
        else:
            return self.model(input)


# Defines the submodule with skip connection.
# X -------------------identity---------------------- X
#   |-- downsampling -- |submodule| -- upsampling --|
class UnetSkipConnectionBlock(nn.Module):
    def __init__(self, outer_nc, inner_nc,
                 submodule=None, outermost=False, innermost=False):
        super(UnetSkipConnectionBlock, self).__init__()
        self.outermost = outermost

        downconv = nn.Conv2d(outer_nc, inner_nc, kernel_size=4,
                             stride=2, padding=1)
        downrelu = nn.LeakyReLU(0.2, True)
        downnorm = nn.BatchNorm2d(inner_nc)
        uprelu = nn.ReLU(True)
        upnorm = nn.BatchNorm2d(outer_nc)

        if outermost:
            upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1)
            down = [downconv]
            up = [uprelu, upconv, nn.Tanh()]
            model = down + [submodule] + up
        elif innermost:
            upconv = nn.ConvTranspose2d(inner_nc, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1)
            down = [downrelu, downconv]
            up = [uprelu, upconv, upnorm]
            model = down + up
        else:
            upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1)
            down = [downrelu, downconv, downnorm]
            up = [uprelu, upconv, upnorm]
            model = down + [submodule] + up

        self.model = nn.Sequential(*model)

    def forward(self, x):
        if self.outermost:
            return self.model(x)
        else:
            return torch.cat([self.model(x), x], 1)


# Defines the PatchGAN discriminator with the specified arguments.
class NLayerDiscriminator(nn.Module):
    def __init__(self, input_nc, ndf=64, n_layers=3, use_sigmoid=False, gpu_ids=[]):
        super(NLayerDiscriminator, self).__init__()
        self.gpu_ids = gpu_ids
        self.input_nc = input_nc
        kw = 4
        sequence = [
            nn.Linear(self.input_nc,64),
            nn.LeakyReLU(0.2, True),
            nn.Linear(64,1),
        ]
#        sequence = [
#            nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=2),
#            nn.LeakyReLU(0.2, True)
#        ]
#
#        nf_mult = 1
#        nf_mult_prev = 1
#        for n in range(1, n_layers):
#            nf_mult_prev = nf_mult
#            nf_mult = min(2**n, 8)
#            sequence += [
#                nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult,
#                                kernel_size=kw, stride=2, padding=2),
#                # TODO: use InstanceNorm
#                nn.BatchNorm2d(ndf * nf_mult),
#                nn.LeakyReLU(0.2, True)
#            ]
#
#        nf_mult_prev = nf_mult
#        nf_mult = min(2**n_layers, 8)
#        sequence += [
#            nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult,
#                            kernel_size=1, stride=2, padding=2),
#            # TODO: useInstanceNorm
#            nn.BatchNorm2d(ndf * nf_mult),
#            nn.LeakyReLU(0.2, True)
#        ]
#
#        sequence += [nn.Conv2d(ndf * nf_mult, 1, kernel_size=1)]
        sequence += [nn.Sigmoid()]

        self.model = nn.Sequential(*sequence)

    def forward(self, input):
        if isinstance(input.data, torch.cuda.FloatTensor) and self.gpu_ids:
            #return nn.parallel.data_parallel(self.model, input.view(1,1,28,28), self.gpu_ids)
            #print(input)
            return self.model(input.view(self.input_nc))
        else:
            return self.model(input.view(self.input_nc))

# Instance Normalization layer from
# https://github.com/darkstar112358/fast-neural-style

class InstanceNormalization(torch.nn.Module):
    """InstanceNormalization
    Improves convergence of neural-style.
    ref: https://arxiv.org/pdf/1607.08022.pdf
    """

    def __init__(self, dim, eps=1e-5):
        super(InstanceNormalization, self).__init__()
        self.weight = nn.Parameter(torch.FloatTensor(dim))
        self.bias = nn.Parameter(torch.FloatTensor(dim))
        self.eps = eps
        self._reset_parameters()

    def _reset_parameters(self):
        self.weight.data.uniform_()
        self.bias.data.zero_()

    def forward(self, x):
        n = x.size(2) * x.size(3)
        t = x.view(x.size(0), x.size(1), n)
        mean = torch.mean(t, 2).unsqueeze(2).expand_as(x)
        # Calculate the biased var. torch.var returns unbiased var
        var = torch.var(t, 2).unsqueeze(2).expand_as(x) * ((n - 1) / float(n))
        scale_broadcast = self.weight.unsqueeze(1).unsqueeze(1).unsqueeze(0)
        scale_broadcast = scale_broadcast.expand_as(x)
        shift_broadcast = self.bias.unsqueeze(1).unsqueeze(1).unsqueeze(0)
        shift_broadcast = shift_broadcast.expand_as(x)
        out = (x - mean) / torch.sqrt(var + self.eps)
        out = out * scale_broadcast + shift_broadcast
        return out
