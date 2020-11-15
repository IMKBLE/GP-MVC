import numpy as np
import torch
import os
from collections import OrderedDict
from pdb import set_trace as st
from torch.autograd import Variable
import itertools
import util.util as util
from util.image_pool import ImagePool
from .base_model import BaseModel
from . import networks
import sys

class VIGANModel(BaseModel):
    def name(self):
        return 'VIGANModel'

    def initialize(self, opt):
        BaseModel.initialize(self, opt)

        nb = opt.batchSize
        size = opt.fineSize
        self.input_A = self.Tensor(nb, opt.input_nc, size, size)
        self.input_B = self.Tensor(nb, opt.output_nc, size, size)
        self.input_C = self.Tensor(nb, opt.output_nc, size, size)
        # load/define networks
        ###################################
        self.clu = networks.define_clustering(10,100,self.gpu_ids)
        self.netG_A = networks.define_G(opt.input_nA, opt.input_nB,
                                     opt.ngf, opt.which_model_netG, opt.norm, self.gpu_ids)
        self.netG_B = networks.define_G(opt.input_nB, opt.input_nA,
                                    opt.ngf, opt.which_model_netG, opt.norm, self.gpu_ids)
        self.AE = networks.define_AE(opt.input_nA, opt.input_nB, opt.input_nCC, self.gpu_ids)
        #self.AE_B = networks.define_AE(128, 144, self.gpu_ids)
        #####################################
        if self.isTrain:
            use_sigmoid = opt.no_lsgan
            self.netD_A = networks.define_D(opt.input_nA, opt.ndf,
                                         opt.which_model_netD,
                                         opt.n_layers_D, use_sigmoid, self.gpu_ids)
            self.netD_B = networks.define_D(opt.input_nB, opt.ndf,
                                         opt.which_model_netD,
                                         opt.n_layers_D, use_sigmoid, self.gpu_ids)
            self.netD_C = networks.define_D(opt.input_nCC, opt.ndf,
                                         opt.which_model_netD,
                                         opt.n_layers_D, use_sigmoid, self.gpu_ids)
        if not self.isTrain or opt.continue_train:
            which_epoch = opt.which_epoch
            self.load_network(self.netG_A, 'G_A', which_epoch)
            self.load_network(self.netG_B, 'G_B', which_epoch)
            self.load_network(self.AE, 'AE', which_epoch)
            #self.load_network(self.AE, 'AE', which_epoch)
            if self.isTrain:
                self.load_network(self.netD_A, 'D_A', which_epoch)
                self.load_network(self.netD_B, 'D_B', which_epoch)
                self.load_network(self.netD_C, 'D_C', which_epoch)

        if self.isTrain:
            self.old_lr = opt.lr
            self.fake_A_pool = ImagePool(opt.pool_size)
            self.fake_B_pool = ImagePool(opt.pool_size)

            # define loss functions
            self.criterionGAN = networks.GANLoss(use_lsgan=not opt.no_lsgan, tensor=self.Tensor)
            self.criterionCycle = torch.nn.L1Loss()
            self.criterionIdt = torch.nn.L1Loss()
            self.criterionAE = torch.nn.MSELoss()
            self.commonloss = torch.nn.KLDivLoss()
            self.criterion = torch.nn.CrossEntropyLoss()
            
            # initialize optimizers
            self.optimizer_G = torch.optim.Adam(self.AE.parameters(),
                  #itertools.chain(self.AE.layer5_1.parameters(),self.AE.layer5_2.parameters(),self.AE.layer6_1.parameters(),self.AE.layer6_2.parameters()),
                  lr=opt.lr, betas=(opt.beta1, 0.999)) 
            self.optimizer_D_A = torch.optim.Adam(self.netD_A.parameters(),
                                                lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_D_B = torch.optim.Adam(self.netD_B.parameters(),
                                                lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_D_C = torch.optim.Adam(self.netD_C.parameters(),
                                                lr=opt.lr, betas=(opt.beta1, 0.999))

            self.optimizer_D_A_AE = torch.optim.Adam(self.netD_A.parameters(),
                                                     lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_D_B_AE = torch.optim.Adam(self.netD_B.parameters(),
                                                     lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_AE = torch.optim.Adam(self.AE.parameters(),
                                                 lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_aes = torch.optim.Adam(self.AE.parameters(),
                                                 lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_aep = torch.optim.Adam(self.AE.parameters(),
                                                 lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_AE2 = torch.optim.Adam(self.AE.parameters(),
                                                 lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_AE_GA_GB = torch.optim.Adam(
                itertools.chain(self.AE.parameters(),self.clu.parameters()),
                lr=opt.lr, betas=(opt.beta1, 0.999)) 
            self.optimizer_AE_CL = torch.optim.Adam(itertools.chain(self.AE.parameters(), self.clu.parameters()),
                                                   lr=opt.lr, betas=(opt.beta1, 0.999))

            print('---------- Networks initialized -------------')
            networks.print_network(self.netG_A)
            networks.print_network(self.netG_B)
            networks.print_network(self.netD_A)
            networks.print_network(self.netD_B)
            networks.print_network(self.netD_C)
            networks.print_network(self.AE)
            print('-----------------------------------------------')

    def set_input(self, images_a, images_b, images_c):
        input_A =images_a
        input_B =images_b
        input_C =images_c

        self.input_A.resize_(input_A.size()).copy_(input_A)
        self.input_B.resize_(input_B.size()).copy_(input_B)
        self.input_C.resize_(input_C.size()).copy_(input_C)
        
    def set_input_train(self, images_a, images_b, images_c, target_i):
        input_A =images_a
        input_B =images_b
        input_C =images_c
        #print('input_A:')
        #print(input_A)
        self.input_A.resize_(input_A.size()).copy_(input_A)
        self.input_B.resize_(input_B.size()).copy_(input_B)
        self.input_C.resize_(input_C.size()).copy_(input_C)
#        self.number_i = number_i
        self.target_i = Variable(target_i)
#        self.center = Variable(centroids, requires_grad=True)
#        self.shared = commonz

    def forward(self):
        self.real_A = Variable(self.input_A)
        self.real_B = Variable(self.input_B)
        self.real_C = Variable(self.input_C)

    def test(self):
        self.real_A = Variable(self.input_A, volatile=True)
        self.fake_B = self.netG_A.forward(self.real_A)
        self.rec_A = self.netG_B.forward(self.fake_B)

        self.real_B = Variable(self.input_B, volatile=True)
        self.fake_A = self.netG_B.forward(self.real_B)
        self.rec_B  = self.netG_A.forward(self.fake_A)

        # Autoencoder loss: fakeA
        self.AEfakeA, AErealB = self.AE.forward(self.fake_A, self.real_B)
        # Autoencoder loss: fakeB
        AErealA, self.AEfakeB = self.AE.forward(self.real_A, self.fake_B)

    def test_unpaired(self):
        self.real_A = Variable(self.input_A, volatile=True)
        self.real_B = Variable(self.input_B, volatile=True)
        self.real_C = Variable(self.input_C, volatile=True)
        
        AErealA1,self.fake_B, AErealC1,self.com1 = self.AE.forward_ac2b(self.real_A,self.real_C)
        
        self.fake_A, AErealB2,AErealC2,self.com2 = self.AE.forward_bc2a(self.real_B,self.real_C)
        
        AErealA3, AErealB3, self.fake_C,self.com3 = self.AE.forward_ab2c(self.real_A,self.real_B)
        
        
        return self.fake_BA, self.fake_AB,self.com1,self.com2
    
    def test_unpaired_a(self):
        self.real_B = Variable(self.input_B, volatile=True)
        self.real_C = Variable(self.input_C, volatile=True)
        
        self.fake_A, AErealB2, AErealC2, self.com2 = self.AE.forward_bc2a(self.real_B,self.real_C)
                
        return self.fake_A
    
    def test_unpaired_b(self):
        self.real_A = Variable(self.input_A, volatile=True)
        self.real_C = Variable(self.input_C, volatile=True)
        
        AErealA1,self.fake_B, AErealC1,self.com1 = self.AE.forward_ac2b(self.real_A,self.real_C)
                
        return self.fake_B
    
    def test_unpaired_c(self):
        self.real_A = Variable(self.input_A, volatile=True)
        self.real_B = Variable(self.input_B, volatile=True)
        
        AErealA3, AErealB3, self.fake_C,self.com3 = self.AE.forward_ab2c(self.real_A,self.real_B)
                
        return self.fake_C

    def test_commonZ(self):
        self.real_A = Variable(self.input_A, volatile=True)
        #self.fake_B = self.netG_A.forward(self.real_A)
        #self.rec_A = self.netG_B.forward(self.fake_B)

        self.real_B = Variable(self.input_B, volatile=True)
        #self.fake_A = self.netG_B.forward(self.real_B)
        #self.rec_B  = self.netG_A.forward(self.fake_A)
        
        self.real_C = Variable(self.input_C, volatile=True)
        
        # Autoencoder loss: fakeA
        self.commonZ = self.AE.forward_commonZ(self.real_A, self.real_B, self.real_C)
        # Autoencoder loss: fakeB
        #AErealA, self.AEfakeB = self.AE.forward(self.real_A, self.real_B)
        return self.commonZ

    def test_2commonZ(self):
        self.real_A = Variable(self.input_A, volatile=True)
        #self.fake_B = self.netG_A.forward(self.real_A)
        #self.rec_A = self.netG_B.forward(self.fake_B)

        self.real_B = Variable(self.input_B, volatile=True)
        #self.fake_A = self.netG_B.forward(self.real_B)
        #self.rec_B  = self.netG_A.forward(self.fake_A)
        
        self.real_C = Variable(self.input_C, volatile=True)
        
        # Autoencoder loss: fakeA
        self.com12, self.com13, self.com23 = self.AE.forward_2common(self.real_A, self.real_B, self.real_C)
        # Autoencoder loss: fakeB
        #AErealA, self.AEfakeB = self.AE.forward(self.real_A, self.real_B)
        return self.com12, self.com13, self.com23
    
    #get image pathss
    def get_image_paths(self):
        return self.image_paths

    def backward_D_basic(self, netD, real, fake):
        # Real
        pred_real = netD.forward(real)
        loss_D_real = self.criterionGAN(pred_real, True)
        # Fake
        pred_fake = netD.forward(fake.detach())
        loss_D_fake = self.criterionGAN(pred_fake, False)
        # Combined loss
        loss_D = (loss_D_real + loss_D_fake) * 0.5
        # backward
        loss_D.backward()
        return loss_D



    def backward_D_A(self):
        fake_A = self.fake_A_pool.query(self.fake_A)
        self.loss_D_A =  self.backward_D_basic(self.netD_A, self.real_A, fake_A)

    def backward_D_B(self):
        fake_B = self.fake_B_pool.query(self.fake_B)
        self.loss_D_B = self.backward_D_basic(self.netD_B, self.real_B, fake_B)


    def backward_D_C(self):
        fake_C = self.fake_A_pool.query(self.fake_C)
        self.loss_D_C =  self.backward_D_basic(self.netD_C, self.real_C, fake_C)        

    ############################################################################
    # Define backward function for VIGAN
    ############################################################################
    def backward_AE_pretrain(self):
        # Autoencoder loss
        
        self.fake_A, AErealB1,AErealC1,self.com1 = self.AE.forward_bc2a(self.real_B,self.real_C)
        
        AErealA2,self.fake_B, AErealC2,self.com2 = self.AE.forward_ac2b(self.real_A,self.real_C)       
        
        AErealA3, AErealB3, self.fake_C,self.com3 = self.AE.forward_ab2c(self.real_A,self.real_B)
#        comz = (self.com1 + self.com2)/2
#        comz = comz.data
#        self.lossz = self.commonloss(self.com1,comz)+self.commonloss(self.com2,comz)
        
        self.loss_AE_A = self.criterionAE(self.fake_A, self.real_A) + self.criterionAE(AErealB1, self.real_B) + self.criterionAE(AErealC1, self.real_C) 
        
        self.loss_AE_B = self.criterionAE(AErealA2, self.real_A) + self.criterionAE(self.fake_B, self.real_B) + self.criterionAE(AErealC2, self.real_C) 

        self.loss_AE_C = self.criterionAE(self.fake_C, self.real_C) + self.criterionAE(AErealB3, self.real_B) +self.criterionAE(AErealA3, self.real_A) 
        self.loss_AE_pre = self.loss_AE_A + self.loss_AE_B + self.loss_AE_C 
        self.loss_AE_pre.backward()


    def backward_AE(self):
        # Autoencoder loss
        lambda_idt = self.opt.identity
        lambda_A = self.opt.lambda_A
        lambda_B = self.opt.lambda_B
        lambda_C = self.opt.lambda_C
  
        self.fake_A, AErealB1, AErealC1,self.com1 = self.AE.forward_bc2a(self.real_B,self.real_C)
        pred_fake = self.netD_A.forward(self.fake_A)
        self.loss_G_A = self.criterionGAN(pred_fake, True)
        
        # D_A(G_A(A))
        AErealA2,self.fake_B, AErealC2,self.com2 = self.AE.forward_ac2b(self.real_A,self.real_C)
        pred_fake = self.netD_B.forward(self.fake_B)
        self.loss_G_B = self.criterionGAN(pred_fake, True)

        # D_C(G_C(C))        
        AErealA3, AErealB3, self.fake_C,self.com3 = self.AE.forward_ab2c(self.real_A,self.real_B)
        pred_fake = self.netD_C.forward(self.fake_C)
        self.loss_G_C = self.criterionGAN(pred_fake, True)
        
        # Forward cycle loss
        self.rec_A1,self.rec_B1,self.rec_C1,self.com12= self.AE.forward_bc2a(self.fake_B,self.fake_C)
        self.loss_cycle_A1 = self.criterionCycle(self.rec_A1, self.real_A) * lambda_A
        self.loss_cycle_B1 = self.criterionCycle(self.rec_B1, self.real_B) * lambda_B
        self.loss_cycle_C1 = self.criterionCycle(self.rec_C1, self.real_C) * lambda_C   
        self.loss_cyc1 = self.loss_cycle_A1 + self.loss_cycle_B1 + self.loss_cycle_B1
        
        # Backward cycle loss
        self.rec_A2,self.rec_B2,self.rec_C2,self.com22 = self.AE.forward_ac2b(self.fake_A,self.fake_C)
        self.loss_cycle_B2 = self.criterionCycle(self.rec_B2, self.real_B) * lambda_B
        self.loss_cycle_A2 = self.criterionCycle(self.rec_A2, self.real_A) * lambda_A
        self.loss_cycle_C2 = self.criterionCycle(self.rec_C2, self.real_C) * lambda_C   
        self.loss_cyc2 = self.loss_cycle_A2 + self.loss_cycle_B2 + self.loss_cycle_B2 
          
        # Backward cycle loss
        self.rec_A3,self.rec_B3,self.rec_C3,self.com32 = self.AE.forward_ab2c(self.fake_A,self.fake_B)
        self.loss_cycle_B3 = self.criterionCycle(self.rec_B3, self.real_B) * lambda_B
        self.loss_cycle_A3 = self.criterionCycle(self.rec_A3, self.real_A) * lambda_A
        self.loss_cycle_C3 = self.criterionCycle(self.rec_C3, self.real_C) * lambda_C          
        self.loss_cyc3 = self.loss_cycle_A3 + self.loss_cycle_B3 + self.loss_cycle_B3
        
        # combined loss
        self.loss_GABC = self.loss_G_A + self.loss_G_B + self.loss_G_C
        
        self.loss_AE_A = self.criterionAE(AErealB1, self.real_B) + self.criterionAE(AErealC1, self.real_C) 
        
        self.loss_AE_B = self.criterionAE(AErealA2, self.real_A) + self.criterionAE(AErealC2, self.real_C) 

        self.loss_AE_C = self.criterionAE(AErealB3, self.real_B) +self.criterionAE(AErealA3, self.real_A) 
        
        self.loss_AE = self.loss_AE_A + self.loss_AE_B + self.loss_AE_C 
        
        self.loss_G = self.loss_GABC + self.loss_AE + self.loss_cyc1 + self.loss_cyc2+ self.loss_cyc3
        self.loss_G.backward()
        
        
        
        
    def backward_G(self):
        lambda_idt = self.opt.identity
        lambda_A = self.opt.lambda_A
        lambda_B = self.opt.lambda_B
        lambda_C = self.opt.lambda_C
  
        self.fake_A, AErealB1, AErealC1,self.com1 = self.AE.forward_bc2a(self.real_B,self.real_C)
        pred_fake = self.netD_A.forward(self.fake_A)
        self.loss_G_A = self.criterionGAN(pred_fake, True)
        
        # D_A(G_A(A))
        AErealA2,self.fake_B, AErealC2,self.com2 = self.AE.forward_ac2b(self.real_A,self.real_C)
        pred_fake = self.netD_B.forward(self.fake_B)
        self.loss_G_B = self.criterionGAN(pred_fake, True)

        # D_C(G_C(C))        
        AErealA3, AErealB3, self.fake_C,self.com3 = self.AE.forward_ab2c(self.real_A,self.real_B)
        pred_fake = self.netD_C.forward(self.fake_C)
        self.loss_G_C = self.criterionGAN(pred_fake, True)
        
        # Forward cycle loss
        self.rec_A1,self.rec_B1,self.rec_C1,self.com12= self.AE.forward_bc2a(self.fake_B,self.fake_C)
        self.loss_cycle_A1 = self.criterionCycle(self.rec_A1, self.real_A) * lambda_A
        self.loss_cycle_B1 = self.criterionCycle(self.rec_B1, self.real_B) * lambda_B
        self.loss_cycle_C1 = self.criterionCycle(self.rec_C1, self.real_C) * lambda_C   
        self.loss_cyc1 = self.loss_cycle_A1 + self.loss_cycle_B1 + self.loss_cycle_B1
        
        # Backward cycle loss
        self.rec_A2,self.rec_B2,self.rec_C2,self.com22 = self.AE.forward_ac2b(self.fake_A,self.fake_C)
        self.loss_cycle_B2 = self.criterionCycle(self.rec_B2, self.real_B) * lambda_B
        self.loss_cycle_A2 = self.criterionCycle(self.rec_A2, self.real_A) * lambda_A
        self.loss_cycle_C2 = self.criterionCycle(self.rec_C2, self.real_C) * lambda_C   
        self.loss_cyc2 = self.loss_cycle_A2 + self.loss_cycle_B2 + self.loss_cycle_B2 
          
        # Backward cycle loss
        self.rec_A3,self.rec_B3,self.rec_C3,self.com32 = self.AE.forward_ab2c(self.fake_A,self.fake_B)
        self.loss_cycle_B3 = self.criterionCycle(self.rec_B3, self.real_B) * lambda_B
        self.loss_cycle_A3 = self.criterionCycle(self.rec_A3, self.real_A) * lambda_A
        self.loss_cycle_C3 = self.criterionCycle(self.rec_C3, self.real_C) * lambda_C          
        self.loss_cyc3 = self.loss_cycle_A3 + self.loss_cycle_B3 + self.loss_cycle_B3
        
        # combined loss
        self.loss_GABC = self.loss_G_A + self.loss_G_B + self.loss_G_C
        
        self.loss_AE_A = self.criterionAE(AErealB1, self.real_B) + self.criterionAE(AErealC1, self.real_C) 
        
        self.loss_AE_B = self.criterionAE(AErealA2, self.real_A) + self.criterionAE(AErealC2, self.real_C) 

        self.loss_AE_C = self.criterionAE(AErealB3, self.real_B) +self.criterionAE(AErealA3, self.real_A) 
        
        self.loss_AE = self.loss_AE_A + self.loss_AE_B + self.loss_AE_C 
        
        self.loss_G = self.loss_GABC + 20*self.loss_AE + 10*self.loss_cyc1 + 10*self.loss_cyc2+ 10*self.loss_cyc3
        self.loss_G.backward()
        
#    def backward_aep(self):
#        # Autoencoder loss
#        self.loss_ae = self.criterionAE(self.fake_AA, self.real_A) +self.criterionAE(self.fake_BB, self.real_B)\
#                        +self.criterionAE(self.fake_A, self.real_A) +self.criterionAE(self.fake_B, self.real_B)
#
#        self.loss_ae.backward()
#    def backward_aes(self):
#        # Autoencoder loss
#        self.loss_ae = self.criterionAE(self.fake_AA, self.real_A) +self.criterionAE(self.fake_BB, self.real_B)
#
#        self.loss_ae.backward()
    # input is vector        


    def backward_AE_CL(self):

        # Autoencoder loss
#        AErealA, AErealB,self.com = self.AE.forward(self.real_A, self.real_B)
        lambda_A = self.opt.lambda_A
        lambda_B = self.opt.lambda_B
        lambda_C = self.opt.lambda_C
        
        self.fake_A, AErealB1, AErealC1, self.com1 = self.AE.forward_bc2a(self.real_B,self.real_C)
        
        AErealA2, self.fake_B, AErealC2, self.com2 = self.AE.forward_ac2b(self.real_A,self.real_C)       
        
        AErealA3, AErealB3, self.fake_C, self.com3 = self.AE.forward_ab2c(self.real_A,self.real_B)
        
        self.loss_AE_A = self.criterionAE(AErealB1, self.real_B) + self.criterionAE(AErealC1, self.real_C) 
        
        self.loss_AE_B = self.criterionAE(AErealA2, self.real_A) + self.criterionAE(AErealC2, self.real_C) 

        self.loss_AE_C = self.criterionAE(AErealB3, self.real_B) +self.criterionAE(AErealA3, self.real_A) 
        
        self.loss_AE = self.loss_AE_A + self.loss_AE_B + self.loss_AE_C 
        
        self.com = self.AE.forward_commonZ(self.real_A, self.real_B, self.real_C)
        # Clustering_loss
        self.q_i,self.q= self.clu.forward(self.com)
        self.loss_com =  self.commonloss(self.q_i,self.target_i.view(1,10))
        
        # Forward cycle loss
        self.rec_A1,self.rec_B1,self.rec_C1,self.com12= self.AE.forward_bc2a(self.fake_B,self.fake_C)
        self.loss_cycle_A1 = self.criterionCycle(self.rec_A1, self.real_A) * lambda_A
        self.loss_cycle_B1 = self.criterionCycle(self.rec_B1, self.real_B) * lambda_B
        self.loss_cycle_C1 = self.criterionCycle(self.rec_C1, self.real_C) * lambda_C   
        self.loss_cyc1 = self.loss_cycle_A1 + self.loss_cycle_B1 + self.loss_cycle_B1
        
        # Backward cycle loss
        self.rec_A2,self.rec_B2,self.rec_C2,self.com22 = self.AE.forward_ac2b(self.fake_A,self.fake_C)
        self.loss_cycle_B2 = self.criterionCycle(self.rec_B2, self.real_B) * lambda_B
        self.loss_cycle_A2 = self.criterionCycle(self.rec_A2, self.real_A) * lambda_A
        self.loss_cycle_C2 = self.criterionCycle(self.rec_C2, self.real_C) * lambda_C   
        self.loss_cyc2 = self.loss_cycle_A2 + self.loss_cycle_B2 + self.loss_cycle_B2 
          
        # Backward cycle loss
        self.rec_A3,self.rec_B3,self.rec_C3,self.com32 = self.AE.forward_ab2c(self.fake_A,self.fake_B)
        self.loss_cycle_B3 = self.criterionCycle(self.rec_B3, self.real_B) * lambda_B
        self.loss_cycle_A3 = self.criterionCycle(self.rec_A3, self.real_A) * lambda_A
        self.loss_cycle_C3 = self.criterionCycle(self.rec_C3, self.real_C) * lambda_C          
        self.loss_cyc3 = self.loss_cycle_A3 + self.loss_cycle_B3 + self.loss_cycle_B3
#        self.fake_AA,self.fake_B,self.com1 = self.AE.forward_a2b(self.real_A)
#        
#        self.fake_A,self.fake_BB,self.com2 = self.AE.forward_b2a(self.real_B)
#                # Clustering_loss
#        self.q_i1,self.q1= self.clu.forward(self.com1)
#        self.loss_clustering1 =  self.commonloss(self.q_i1,self.target_i.view(1,10))
#        
#        self.q_i2,self.q2= self.clu.forward(self.com2)
#        self.loss_clustering2 =  self.commonloss(self.q_i2,self.target_i.view(1,10))
        
        self.loss_clustering = 15*self.loss_com + 10*self.loss_AE+ 1*self.loss_cyc1 + 1*self.loss_cyc2+ 1*self.loss_cyc3#+ 10*self.loss_cycle_A2 + 10*self.loss_cycle_B1 #+self.loss_clustering1 + self.loss_clustering2 
        
        self.loss_AE_CL = self.loss_clustering
        self.loss_AE_CL.backward()


    def backward_D_A_AE(self):
        fake_B = self.AEfakeB
        self.loss_D_A_AE = self.backward_D_basic(self.netD_A, self.real_B, fake_B)

    def backward_D_B_AE(self):
        fake_A = self.AEfakeA
        self.loss_D_B_AE =  self.backward_D_basic(self.netD_B, self.real_A, fake_A)


    def backward_AE_GA_GB(self):
        
        lambda_C = self.opt.lambda_C
        lambda_D = self.opt.lambda_D

        # fake data
        # G_A(A)
        self.fake_B = self.netG_A.forward(self.real_A)
        # G_B(B)
        self.fake_A = self.netG_B.forward(self.real_B)

        # Forward cycle loss
        self.rec_A = self.netG_B.forward(self.fake_B)
        self.loss_cycle_A_AE = self.criterionCycle(self.rec_A, self.real_A)
        # Backward cycle loss
        self.rec_B = self.netG_A.forward(self.fake_A)
        self.loss_cycle_B_AE = self.criterionCycle(self.rec_B, self.real_B)

        # Autoencoder loss: fakeA
        self.AEfakeA, AErealB,self.com1 = self.AE.forward(self.fake_A, self.real_B)
        self.loss_AE_fA_rB = (self.criterionAE(self.AEfakeA, self.real_A) + self.criterionAE(AErealB, self.real_B)) * 1

        # Autoencoder loss: fakeB
        AErealA, self.AEfakeB,self.com2 = self.AE.forward(self.real_A, self.fake_B)
        self.loss_AE_rA_fB = (self.criterionAE(AErealA, self.real_A) + self.criterionAE(self.AEfakeB, self.real_B)) * 1
        self.loss_AE = (self.loss_AE_fA_rB + self.loss_AE_rA_fB)
        
        # Clustering_loss
        self.q_i,self.q= self.clu.forward(self.com1)
        self.loss_clustering1 =  self.commonloss(self.q_i,self.target_i.view(1,10))
        
        self.q_i2,self.q2= self.clu.forward(self.com2)
        self.loss_clustering2 =  self.commonloss(self.q_i2,self.target_i.view(1,10))
        
        self.clustering_loss = self.loss_clustering1 +self.loss_clustering2
        # D loss
        pred_fake = self.netD_A.forward(self.AEfakeB)
        self.loss_AE_GA = self.criterionGAN(pred_fake, True)
        pred_fake = self.netD_B.forward(self.AEfakeA)
        self.loss_AE_GB = self.criterionGAN(pred_fake, True)

        self.loss_AE_GA_GB =lambda_C * ( self.loss_AE_GA + self.loss_AE_GB) + \
                             lambda_D * self.loss_AE + 2 * (self.loss_cycle_A_AE + self.loss_cycle_B_AE) +\
                            5*self.loss_clustering1 + 5*self.loss_clustering2 
        self.loss_AE_GA_GB.backward()




    ############################################################################
    # Define optimize function for VIGAN
    ############################################################################
    def optimize_parameters_pretrain_AE(self):
        # forward
        self.forward()
        
#        for i in range(1):
        # AE
        self.optimizer_AE.zero_grad()
        self.backward_AE_pretrain()
        self.optimizer_AE.step()

#        for i in range(1):
#            # D_A
#            self.optimizer_D_A.zero_grad()
#            self.backward_D_A()
#            self.optimizer_D_A.step()
#            # D_B
#            self.optimizer_D_B.zero_grad()
#            self.backward_D_B()
#            self.optimizer_D_B.step()
    def optimize_parameters_AE(self):
        # forward
        self.forward()
        
#        for i in range(1):
        # AE
        self.optimizer_AE2.zero_grad()
        self.backward_AE()
        self.optimizer_AE2.step()

        for i in range(1):
            # D_A
            self.optimizer_D_A.zero_grad()
            self.backward_D_A()
            self.optimizer_D_A.step()
            # D_B
            self.optimizer_D_B.zero_grad()
            self.backward_D_B()
            self.optimizer_D_B.step()
    #########################################################################################################
    def optimize_parameters_pretrain_cycleGAN(self):
        # forward
        self.forward()
        # G_A and G_B
        

        self.optimizer_G.zero_grad()
        self.backward_G()
        self.optimizer_G.step()

#        self.optimizer_aes.zero_grad()
#        self.backward_aes()
#        self.optimizer_aes.step()
#        for i in range(1):
        # D_A
        self.optimizer_D_A.zero_grad()
        self.backward_D_A()
        self.optimizer_D_A.step()
        # D_B
        self.optimizer_D_B.zero_grad()
        self.backward_D_B()
        self.optimizer_D_B.step()

        self.optimizer_D_C.zero_grad()
        self.backward_D_C()
        self.optimizer_D_C.step()
        
#    def optimize_parameters_pretrain_GAN(self):
#        # forward
#        self.forward()
#        # G_A and G_B
#        
#
#        self.optimizer_G.zero_grad()
#        self.backward_G()
#        self.optimizer_G.step()
#        
#        self.optimizer_aep.zero_grad()
#        self.backward_aep()
#        self.optimizer_aep.step()
##        for i in range(1):
#        # D_A
#        self.optimizer_D_A.zero_grad()
#        self.backward_D_A()
#        self.optimizer_D_A.step()
#        # D_B
#        self.optimizer_D_B.zero_grad()
#        self.backward_D_B()
#        self.optimizer_D_B.step()
            
    def optimize_AECL(self):
        # forward
        self.forward()

        self.optimizer_AE_CL.zero_grad()
        self.backward_AE_CL()
        self.optimizer_AE_CL.step()
        
#        for i in range(1):
#            # D_A
#            self.optimizer_D_A.zero_grad()
#            self.backward_D_A()
#            self.optimizer_D_A.step()
#            # D_B
#            self.optimizer_D_B.zero_grad()
#            self.backward_D_B()
#            self.optimizer_D_B.step()
    def optimize_parameters(self):
        # forward

        self.forward()
#        self.optimizer_AE_GA_GB.zero_grad()
#        self.backward_AE_GA_GB()
#        self.optimizer_AE_GA_GB.step()
        # AE+G_A+G_B
        
        for i in range(2):
            self.optimizer_AE_GA_GB.zero_grad()
            self.backward_AE_GA_GB()
            self.optimizer_AE_GA_GB.step()

        for i in range(1):
            # D_A
            self.optimizer_D_A_AE.zero_grad()
            self.backward_D_A_AE()
            self.optimizer_D_A_AE.step()
            # D_B
            self.optimizer_D_B_AE.zero_grad()
            self.backward_D_B_AE()
            self.optimizer_D_B_AE.step()

    ############################################################################################
    # Get errors for visualization
    ############################################################################################
    def get_current_errors_AE_pre(self):
        AE = self.loss_AE_pre.data[0]
        AE_A = self.loss_AE_A.data[0]
        AE_B = self.loss_AE_B.data[0]
        if self.opt.identity > 0.0:
            return OrderedDict([('AE', AE), ('AE_A', AE_A), ('AE_B', AE_B)])
        else:
            return OrderedDict([('AE', AE), ('AE_A', AE_A), ('AE_B', AE_B)])
    
    def get_current_errors_AE(self):
        AE_D_A = self.loss_D_A.data[0]
        AE_G_A = self.loss_G_A.data[0]
#        Cyc_A = self.loss_cycle_A.data[0]
        AE_D_B = self.loss_D_B.data[0]
        AE_G_B = self.loss_G_B.data[0]
#        Cyc_B = self.loss_cycle_B.data[0]
        AE = self.loss_AE.data[0]
#        CLU_loss = self.loss_clustering.data[0]
        ALL_loss = self.loss_G.data[0]
        if self.opt.identity > 0.0:
            idt_A = self.loss_idt_A.data[0]
            idt_B = self.loss_idt_B.data[0]
            return OrderedDict([('D_A', AE_D_A), ('G_A', AE_G_A), ('Cyc_A', Cyc_A), ('idt_A', idt_A),
                                ('D_B', AE_D_B), ('G_B', AE_G_B), ('Cyc_B', Cyc_B), ('idt_B', idt_B)])
        else:
#            return OrderedDict([('D_A', AE_D_A), ('G_A', AE_G_A), ('Cyc_A', Cyc_A),
#                                ('D_B', AE_D_B), ('G_B', AE_G_B), ('Cyc_B', Cyc_B)])
            return OrderedDict([('D_A', AE_D_A), ('G_A', AE_G_A), ('AE', AE),
                                ('D_B', AE_D_B), ('G_B', AE_G_B),  ('ALL', ALL_loss)])
        
    def get_current_errors_AE_CL(self):
#        AE = self.loss_AE.data[0]
        CLU_loss = self.loss_clustering.data[0]
        AE = self.loss_AE.data[0]
        if self.opt.identity > 0.0:
            return OrderedDict([('CLU', CLU_loss), ('AE', AE)])
        else:
            return OrderedDict([ ('CLU', CLU_loss), ('AE', AE)])
        
    def get_current_errors_cycle(self):
        AE_D_A = self.loss_D_A.data[0]
        AE_G_A = self.loss_G_A.data[0]
#        Cyc_A = self.loss_cycle_A.data[0]
        AE_D_B = self.loss_D_B.data[0]
        AE_G_B = self.loss_G_B.data[0]
#        Cyc_B = self.loss_cycle_B.data[0]
        AE = self.loss_AE.data[0]
#        CLU_loss = self.loss_clustering.data[0]
        ALL_loss = self.loss_G.data[0]
        if self.opt.identity > 0.0:
            idt_A = self.loss_idt_A.data[0]
            idt_B = self.loss_idt_B.data[0]
            return OrderedDict([('D_A', AE_D_A), ('G_A', AE_G_A), ('Cyc_A', Cyc_A), ('idt_A', idt_A),
                                ('D_B', AE_D_B), ('G_B', AE_G_B), ('Cyc_B', Cyc_B), ('idt_B', idt_B)])
        else:
#            return OrderedDict([('D_A', AE_D_A), ('G_A', AE_G_A), ('Cyc_A', Cyc_A),
#                                ('D_B', AE_D_B), ('G_B', AE_G_B), ('Cyc_B', Cyc_B)])
            return OrderedDict([('D_A', AE_D_A), ('G_A', AE_G_A), ('AE', AE),
                                ('D_B', AE_D_B), ('G_B', AE_G_B),  ('ALL', ALL_loss)])

    def get_current_errors(self):
        D_A = self.loss_D_A_AE.data[0]
        G_A = self.loss_AE_GA.data[0]
        Cyc_A = self.loss_cycle_A_AE.data[0]
        D_B = self.loss_D_B_AE.data[0]
        G_B = self.loss_AE_GB.data[0]
        Cyc_B = self.loss_cycle_B_AE.data[0]
        clu_loss = self.clustering_loss.data[0]
        loss_all = self.loss_AE_GA_GB.data[0]
        if self.opt.identity > 0.0:
            idt_A = self.loss_idt_A.data[0]
            idt_B = self.loss_idt_B.data[0]
            return OrderedDict([('D_A', D_A), ('G_A', G_A), ('Cyc_A', Cyc_A), ('idt_A', idt_A),
                                ('D_B', D_B), ('G_B', G_B), ('Cyc_B', Cyc_B), ('idt_B', idt_B)])
        else:
            return OrderedDict([('D_A', D_A), ('G_A', G_A), ('Cyc_A', Cyc_A),('loss_all',loss_all),
                                ('D_B', D_B), ('G_B', G_B), ('Cyc_B', Cyc_B),('clu_loss',clu_loss)])

    def get_current_visuals(self):
        real_A = util.tensor2im(self.real_A.data)
        fake_B = util.tensor2im(self.fake_B.data)
        rec_A  = util.tensor2im(self.rec_A.data)
        real_B = util.tensor2im(self.real_B.data)
        fake_A = util.tensor2im(self.fake_A.data)
        rec_B  = util.tensor2im(self.rec_B.data)

        AE_fake_A = util.tensor2im(self.AEfakeA.view(1,1,28,28).data)
        AE_fake_B = util.tensor2im(self.AEfakeB.view(1,1,28,28).data)


        if self.opt.identity > 0.0:
            idt_A = util.tensor2im(self.idt_A.data)
            idt_B = util.tensor2im(self.idt_B.data)
            return OrderedDict([('real_A', real_A), ('fake_B', fake_B), ('rec_A', rec_A), ('idt_B', idt_B),
                                ('real_B', real_B), ('fake_A', fake_A), ('rec_B', rec_B), ('idt_A', idt_A),
                                ('AE_fake_A', AE_fake_A), ('AE_fake_B', AE_fake_B)])
        else:
            return OrderedDict([('real_A', real_A), ('fake_B', fake_B), ('rec_A', rec_A),
                                ('real_B', real_B), ('fake_A', fake_A), ('rec_B', rec_B),
                                ('AE_fake_A', AE_fake_A), ('AE_fake_B', AE_fake_B)])

    def save(self, label):
        self.save_network(self.netG_A, 'G_A', label, self.gpu_ids)
        self.save_network(self.netD_A, 'D_A', label, self.gpu_ids)
        self.save_network(self.netG_B, 'G_B', label, self.gpu_ids)
        self.save_network(self.netD_B, 'D_B', label, self.gpu_ids)
        self.save_network(self.netD_C, 'D_C', label, self.gpu_ids)
        self.save_network(self.AE, 'AE', label, self.gpu_ids)
        self.save_network(self.clu, 'CLU', label, self.gpu_ids)

    def update_learning_rate(self):
        lrd = self.opt.lr / self.opt.niter_decay
        lr = self.old_lr - lrd
        for param_group in self.optimizer_D_A.param_groups:
            param_group['lr'] = lr
        for param_group in self.optimizer_D_B.param_groups:
            param_group['lr'] = lr
        for param_group in self.optimizer_D_C.param_groups:
            param_group['lr'] = lr
        for param_group in self.optimizer_G.param_groups:
            param_group['lr'] = lr

        print('update learning rate: %f -> %f' % (self.old_lr, lr))
        self.old_lr = lr
