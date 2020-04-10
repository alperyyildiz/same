import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pickle
import time
import shutil
import os
import hyperopt
from datetime import datetime
from torch import optim
from sklearn import datasets
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset
from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.seasonal import seasonal_decompose, STL, DecomposeResult
from sklearn.model_selection import GridSearchCV
from hyperopt import fmin, tpe, hp, STATUS_OK, base,Trials
cuda = torch.device('cuda')


class PARAMETERS():
    
    def __init__(self):
        super().__init__()
        self.scaler = StandardScaler()

    def DICT_TO_LIST(self):
        prev_out_ch = 0
        self.LIST = list()
        self.seq_len_left = self.DICT['OTHERS']['1']['windowlength']
        for tt, key in enumerate(list(self.DICT.keys())):
            if key == 'flatten':
                self.LIST.append([key, ['nothing','nothing']])
                prev_out_ch = prev_out_ch * self.seq_len_left
            elif key != 'OTHERS':
                for ttt, layer in enumerate(list(self.DICT[key].keys())):
                    act = False
                    p_list = list()
                    for tttt, param in enumerate(list(self.DICT[key][layer].keys())):
                        if param not in ['dropout','batchnorm','activation_function','pooling']:
                            if param == 'KER':
                                self.seq_len_left = self.seq_len_left - (self.DICT[key][layer][param]-1)*self.DICT[key][layer]['dilation'] 
                            if tt is 0 and ttt is 0:
                                if tttt is 0:
                                    p_list.append(self.featuresize)       
                                p_list.append(self.DICT[key][layer][param])
                            else:
                                if tttt is 0:
                                    p_list.append(prev_out_ch)       
                                p_list.append(self.DICT[key][layer][param])
                    if key == 'LSTM':
                        self.LSTM_PARAMS = p_list
                        print('lstm parameters are {}'.format(p_list))
                    self.LIST.append([key, p_list])
                    prev_out_ch = p_list[1]
                    

                    if 'batchnorm' in list(self.DICT[key][layer].keys()):
                        if self.DICT[key][layer]['batchnorm'] is True:
                            self.LIST.append(['batchnorm',[prev_out_ch,True]])
                    if key != 'LSTM':
                        if self.DICT[key][layer]['dropout'][0] is True:
                            self.LIST.append(['dropout', [self.DICT[key][layer]['dropout'][1], False]])
                    
                    if self.DICT[key][layer]['activation_function'][0] is True:
                        self.LIST.append([self.DICT[key][layer]['activation_function'][1],[False,False]])

                    if key == 'CONV':
                        if self.DICT[key][layer]['pooling'][0] is True:
                            self.LIST.append(['pooling',self.DICT[key][layer]['pooling'][1],self.DICT[key][layer]['pooling'][2]])
                        

           
    def GET_OTHERS(self,OTHERS=None):
        if OTHERS is None:
            OTHERS  =  {
                            'windowlength': 24,
                            'out_size': 4,
                            'period': 24,
                            'lrate': 0.003,
                            'batchsize': 16,
                            'epoch': 550
                            }
        return OTHERS
        
    def GET_DICT(self,DICT=None):
        #USES: GET_OTHERS


        #CREATES: self.DICT

        try:
            _ = list(DICT.keys())
            if len(_)>0:
                self.DICT = DICT
        except:
    
    
    
            OTHERS = self.GET_OTHERS()
            self.DICT = {'CONV': {
                                    '1': {'FIL': 64, 
                                          'KER': 4,
                                          'dropout': [True, 0.5],
                                          'batchnorm': False,
                                          'activation_function': [True, 'relu'],
                                          'pooling': [False, 0, None]
                                        },
                                    '2': {'FIL': 32, 
                                          'KER': 4,
                                          'dropout': [True, 0.5],
                                          'batchnorm': False,
                                          'activation_function': [True, 'relu'],
                                          'pooling': [True, 2, None]
                                        }
                                  },
                        'LSTM': {
                                    '1': {'FIL': 64, 
                                          'dropout': [True, 0.5],
                                          'batchnorm': False,
                                          'activation_function': [True, 'relu'],
                                          'num_of_directions': 1
                                        }
                        },

                      'flatten': {'1': {'nofilter':0 , 'nonothing':0 }},
            
                      'DENSE': {
                      
                                '1': {'FIL': 64,
                                      'dropout' : [True,0.6],
                                      'activation_function': [True, 'relu']
                                    },

                                '2': {'FIL':OTHERS['out_size'],
                                      'dropout' : [False,0],
                                      'activation_function': [False, '-']
                                      }
                              },
                        
                      'OTHERS': {'1':OTHERS}
            }
            
    def CREATE_SEARCH_SPACE(self,TO_CHNG= None):
        #USES: GET_PARAMS_TO_CHANGE()

        #CREATES: self.space

        space = {}
        self.PARAMS_TO_CHANGE = self.GET_PARAMS_TO_CHANGE()
        HYP_DICT ={}
        d_count = 0
        for TYPE in  list(self.PARAMS_TO_CHANGE.keys()):
            HYP_DICT_LAYER = {}
            for layernum in list(self.PARAMS_TO_CHANGE[TYPE].keys()):
                HYP_DICT_PARAMS = {}
                for PARAM in list(self.PARAMS_TO_CHANGE[TYPE][layernum].keys()):
                    if PARAM == 'dropout':
                        d_count = d_count + 1
                        HYP_DICT_PARAMS[PARAM + str(d_count)] = hp.uniform(PARAM + str(d_count),self.PARAMS_TO_CHANGE[TYPE][layernum][PARAM][0],self.PARAMS_TO_CHANGE[TYPE][layernum][PARAM][1])
                    else:
                        HYP_DICT_PARAMS[PARAM + layernum] = hp.uniform(PARAM + layernum,self.PARAMS_TO_CHANGE[TYPE][layernum][PARAM][0],self.PARAMS_TO_CHANGE[TYPE][layernum][PARAM][1])
                    HYP_DICT_LAYER[layernum] =  HYP_DICT_PARAMS
                HYP_DICT[TYPE] =  HYP_DICT_LAYER

        self.space = hp.choice('paramz', [HYP_DICT])

    def GET_PARAMS_TO_CHANGE(self,PARAMS_TO_CHANGE=None):
        if PARAMS_TO_CHANGE is None:
              
            PARAMS_TO_CHANGE = {'CONV': {
                                          '1': {
                                                
                                                'dropout': (0.2, 0.8)
                                              },
                                          '2': {
                                                
                                                'dropout': (0.2, 0.8)
                                              }
                                        },
                                
                              'DENSE': {

                                          '1': {
                                                'dropout' : (0.2,0.8)
                                                }
                                          },
                              'OTHERS':{'1':{'lrate':(0.001,0.0001),
                                            }
                                       }
                              }
        return PARAMS_TO_CHANGE

    #CREATE SUBDIR OF ABOVE NAMED WITH 
    #EXPERIMENT DATE AND START TIME
    def CREATE_DIR(self):
        #INPUT: self.DICT

        first_Con = True
        SAVE_DIR = ''
        repopath = os.getcwd()
        try:
            os.mkdir(repopath + '/storage')
        except:
            pass
        for KEY in list(self.DICT.keys()):
            if KEY != 'OTHERS':
                SAVE_DIR = SAVE_DIR + '_' +KEY + '-'
                for LAYER in list(self.DICT[KEY].keys()):
                    SAVE_DIR = SAVE_DIR + LAYER + '-'
        SAVE_DIR = repopath + '/storage/' + SAVE_DIR
        try: 
            os.mkdir(SAVE_DIR)
            self.SAVE_DIR = SAVE_DIR
        except:
            pass

        SAVE_DIR = SAVE_DIR + '/' +str(datetime.now())[:-10]
        self.SAVE_DIR = SAVE_DIR

        SAVE_DIR_PLOTS = SAVE_DIR + '/' + 'PLOTS'
        self.SAVE_DIR_PLOTS = SAVE_DIR_PLOTS

        SAVE_DIR_MODELS = SAVE_DIR + '/' + 'MODELS'
        self.SAVE_DIR_MODELS = SAVE_DIR_MODELS

        try:
            os.mkdir(SAVE_DIR)
        except:
            pass

        try:
            os.mkdir(SAVE_DIR_PLOTS)
            os.mkdir(SAVE_DIR_MODELS)
        except:
            pass

    #CREATES SAVE NAME FOR BOTH PLOTS
    #AND THE KEY FOR HIST PLOT
    #key_VAR = CHANGING VARS WITH VALUES
    def CREATE_SAVE_NAME(self,DDD):
        save_DIR = ''
        plot_header = ''
        for keynum, KEY in enumerate(list(DDD.keys())):

            save_DIR = save_DIR + '_' + KEY  
            plot_header = plot_header + '\n' + KEY + '--- ' 
            for LAYER in list(DDD[KEY].keys()):
                save_DIR = save_DIR + '-' + LAYER  
                plot_header = plot_header + LAYER + ': ' 
                for PARAM in list(DDD[KEY][LAYER].keys()):
                    save_DIR = save_DIR + PARAM + '-' + str(DDD[KEY][LAYER][PARAM])[:5] + '---'
                    plot_header = plot_header + PARAM + '=' + str(DDD[KEY][LAYER][PARAM])[:5] + '   '
            plot_header = plot_header + '\n'
        return save_DIR, plot_header
    
    #SAVE CONSTANT HYPERPARAMETERS OF EXPERIMENT AS TXT
    def WRITE_CONSTANTS(self):
        key_CONST = ''
        key_VAR = ''
        for KEY in list(self.DICT.keys()):
            exist = True
            if KEY is not 'flatten':
                if KEY not in list(self.PARAMS_TO_CHANGE.keys()):
                    exist = False
                    key_CONST = key_CONST + '\n\n TYPE:   {} \n \n'.format(KEY)
                    key_CONST =  key_CONST + 'LAYER:'
                    for LAYER in list(self.DICT[KEY].keys()):
                        key_CONST =  key_CONST + '\n{} \t---\t'.format(LAYER)
                        for PARAM in list(self.DICT[KEY][LAYER].keys()):
                            key_CONST =  key_CONST + '{}: {} \t\t'.format(PARAM,self.DICT[KEY][LAYER][PARAM])
                        key_CONST = key_CONST + '\n'
                else:
                    key_VAR = key_VAR + '\n\n TYPE:   {}\n\n'.format(KEY)
                    key_VAR = key_VAR + 'LAYER:'

                    key_CONST = key_CONST + '\n\n TYPE:   {} \n \n'.format(KEY)
                    key_CONST =  key_CONST + 'LAYER:'

                    for LAYER in list(self.DICT[KEY].keys()):
                        key_CONST =  key_CONST + '\n{} \t---\t'.format(LAYER)
                        if LAYER in list(self.PARAMS_TO_CHANGE[KEY].keys()):
                            key_VAR =  key_VAR + '\n{} \t---\t'.format(LAYER)
                            for PARAM in list(self.DICT[KEY][LAYER].keys()):
                                if PARAM in list(self.PARAMS_TO_CHANGE[KEY][LAYER].keys()):
                                    key_VAR =  key_VAR + '{}: {} \t\t'.format(PARAM,self.PARAMS_TO_CHANGE[KEY][LAYER][PARAM])
                                else:
                                    key_CONST =  key_CONST + '{}: {} \t\t'.format(PARAM,self.DICT[KEY][LAYER][PARAM])

                        else:
                            for PARAM in list(self.DICT[KEY][LAYER].keys()):
                                key_CONST =  key_CONST + '{}: {} \t\t'.format(PARAM,self.DICT[KEY][LAYER][PARAM])

                     
        with open(self.SAVE_DIR + '/CONSTANT_HYPERPARAMETERS.txt' , 'w') as file: 
            file.write(key_CONST)
        with open(self.SAVE_DIR + '/PARAMETERS_TO_TUNE.txt', 'w') as file:
            file.write(key_VAR) 
 

    #SAVES HIST AND PRED PLOTS
    def SAVE_PLOTS(self,DDD):
        save_NAME, plot_header = self.CREATE_SAVE_NAME(DDD)
        fig = plt.figure(figsize=(12,6))
        fig.suptitle(plot_header)
        plt.plot(self.hist)
        plt.plot(self.hist_valid)
        plt.plot(np.full(shape=(np.array(self.hist).shape[0]),fill_value=0.05),'--r')
        plt.plot(np.full(shape=(np.array(self.hist).shape[0]),fill_value=0.8),'--b')
        #plt.ylim(()
        plt.savefig( self.SAVE_DIR + '/PLOTS/' + save_NAME + '.png')
        
        fig2 = self.plotz()
        fig2.suptitle(plot_header)
        plt.savefig(self.SAVE_DIR + '/PLOTS/PRD_' + save_NAME + '.png')
        plt.close('all')

    def CONV_DICT_TO_INT(self,DDD):
        for KEY in list(DDD.keys()):
            for LAYER in list(DDD[KEY].keys()):
                for PARAM in list(DDD[KEY][LAYER].keys()):    
                    if LAYER != 'flatten':
                        if PARAM[:7] != 'dropout':
                            if PARAM[:9] != 'batchnorm':
                                if PARAM[:5] != 'lrate':
                                    DDD[KEY][LAYER][PARAM] = np.int(np.round(DDD[KEY][LAYER][PARAM]))
        return DDD                



    def preprocess(self):
        global SCALERR

        data = pd.read_excel('clean.xlsx').dropna()
        windowlength = self.DICT['OTHERS']['1']['windowlength']
        period = self.DICT['OTHERS']['1']['period']
        outsize = self.DICT['OTHERS']['1']['out_size']
        batchsize = self.DICT['OTHERS']['1']['batchsize']
        MAX_window = windowlength
        remain_for_train = data.shape[1] - batchsize - windowlength - outsize
        cut_from_begining = (remain_for_train - windowlength - outsize) % batchsize


        arr = np.asarray(data['sales'])
        vv =pd.read_csv('vix.csv',sep=',')
        dol =pd.read_csv('dollar.csv',sep=',')
        vix_inv = np.array(vv['Price'])
        dollar_inv = np.array(dol['Price'])

        #SORT INVERSED
        vix = np.zeros(len(vix_inv))
        for i in range(len(vix)):
            vix[len(vix_inv) - i-1] = vix_inv[i]
        dollars = np.zeros(len(dollar_inv)-1)
        for i in range(1,len(dollars)):
            dollars[len(dollar_inv) - i-1] = dollar_inv[i]

        res = STL(arr,period = period ,seasonal = 23 , trend = 25).fit()

        dataz = np.swapaxes(np.array([res.seasonal,res.trend,res.resid,vix,dollars]),0,1)

        scaler = StandardScaler()
        scaler = scaler.fit(dataz[:-batchsize])
        dataz = scaler.transform(dataz)
        del scaler

        SCALERR = StandardScaler()
        SCALERR = SCALERR.fit(arr[:-batchsize].reshape(-1,1))
        arr = SCALERR.transform(arr.reshape(-1,1))
        arr = arr.reshape(arr.shape[0])

        for feat in range(dataz.shape[1]):
            if feat == 0:
                sliced_windows = np.array([[[ np.array(dataz[:,feat])[i+k+MAX_window-windowlength] for i in  range(windowlength)] for t in range(1)] for k in range(dataz.shape[0] - outsize - MAX_window)])
                results = np.array([[arr[i+k+windowlength] for i in  range(outsize)] for k in range(dataz.shape[0] - outsize - MAX_window)])

            else:
                sliced_new = np.array([[[ np.array(dataz[:,feat])[i+k+MAX_window-windowlength] for i in  range(windowlength)] for t in range(1)] for k in range(dataz.shape[0] - outsize - MAX_window)])
                sliced_windows = np.concatenate((sliced_windows,sliced_new),axis=1)

        TRAIN_IN = sliced_windows[:-24]
        TEST_IN = sliced_windows[-24:]

        TRAIN_OUT = results[:-24]
        TEST_OUT = results[-24:]


        TRAIN_IN = torch.Tensor(TRAIN_IN).to(device = cuda)
        TEST_IN = torch.Tensor(TEST_IN).to(device = cuda)
        TRAIN_OUT = torch.Tensor(TRAIN_OUT).to(device = cuda)
        TEST_OUT = torch.Tensor(TEST_OUT).to(device = cuda)

        TRA_DSet = TensorDataset(TRAIN_IN, TRAIN_OUT)
        VAL_DSet = TensorDataset(TEST_IN, TEST_OUT)

        self.featuresize = TRAIN_IN.shape[1]
        self.TST_INP = TEST_IN
        self.TST_OUT = TEST_OUT
        self.TR_INP = TRAIN_IN
        self.TR_OUT = TRAIN_OUT
        self.train_DL = DataLoader(TRA_DSet, batch_size=batchsize)
        self.val_DL = DataLoader(VAL_DSet, batch_size=batchsize)



    def GET_MODEL(self,DD):
        call_data_again = False
        DD = self.CONV_DICT_TO_INT(DD)    
        for KEY in  list(DD.keys()):
            if KEY is 'OTHERS':
                key__ = list(DD['OTHERS']['1'].keys())
                for key_ in key__:
                    if key_ in ['windowlength','Period','batchsize','outsize']:
                        call_data_again = True
              
            for layernum in list(DD[KEY].keys()):
                for PARAM in list(DD[KEY][layernum].keys()):
                    PARAM_ = PARAM[:-1]
                    if PARAM_ == 'dropout' :
                        self.DICT[KEY][layernum][PARAM_] = [True,DD[KEY][layernum][PARAM]]
                    else:
                        self.DICT[KEY][layernum][PARAM_] = DD[KEY][layernum][PARAM]
        if call_data_again:
            self.preprocess()
        self.DICT_TO_LIST()
        SAVE_NAME, __ = P_OBJ.CREATE_SAVE_NAME(DD)
        self.TRIAL_DIR = self.SAVE_DIR + '/MODELS/' + SAVE_NAME
        try:
            os.mkdir(self.TRIAL_DIR)
        except:
            pass

        model = Model(P_OBJ)
        model.to(device = cuda)
        model.optimizer = optim.Adam(model.parameters(),lr=self.DICT['OTHERS']['1']['lrate'],weight_decay = 0.001 )
        repr(model)

        minloss = model.fit()
        
        model.SAVE_PLOTS(DD)
        
        torch.cuda.empty_cache()
        P_OBJ.EXPERIMENT_NUMBER = P_OBJ.EXPERIMENT_NUMBER + 1
        return {
            'loss': minloss,
            'status': STATUS_OK,
            'attachments':
                {'time_module': pickle.dumps(time.time)}
              }


    def plotz(self):

        timez = np.zeros((350,self.DICT['OTHERS']['1']['out_size']))
        for i in range(350):
            for j in range(self.DICT['OTHERS']['1']['out_size']):
                timez[i,j] = i + j
        inp,out = [self.TST_INP,self.TST_OUT]
        
        self.eval()
        with torch.no_grad():
           pred = self(inp)
        pred1 = pred.cpu()
        out1 = out.cpu()
        pred1 = SCALERR.inverse_transform(pred1)
        out1 = SCALERR.inverse_transform(out1)
        fig = plt.figure(figsize=(12, 6))
        ax1, ax2,  = fig.subplots(2, 1, )
        bisi = out.shape[0]
        for i in range(int(bisi/2)):
            ax1.plot(timez[i],pred1[i])
            ax1.plot(timez[i],out1[i],color='black')

        for i in range(int(bisi/2)+1,2*int(bisi/2)):

            ax2.plot(timez[i],pred1[i])
            ax2.plot(timez[i],out1[i],color='black')

        return fig
class Model(nn.Module, PARAMETERS):
    def __init__(self, SOURCE_OBJ):
        super().__init__()
        self.__dict__.update(SOURCE_OBJ.__dict__)
        self.epoch = self.DICT['OTHERS']['1']['epoch']
        self.layers = nn.ModuleList()
        self.Loss_FUNC = F.mse_loss
        if 'LSTM' in list(self.DICT.keys()):
            self.init_hidden_states()
        for elem in self.LIST:
            key = elem[0]
            args = elem[1]
            self.layer_add(key,*args)
        
    def layer_add(self,key,*args):
        self.layers.append(self.layer_set(key,*args))
        if key == 'LSTM':
            self.init_hidden_states()
        
    def layer_set(self,key,*args):
        ## push args into key layer type, return it
        ## push args into key layer type, return it
        if key == 'CONV':
            return nn.Conv1d(*args)
        elif key == 'LSTM':
            return nn.LSTM(*args)
        elif key == 'DENSE':
            return nn.Linear(*args)
        elif key == 'dropout':
            return nn.Dropout(*args)
        elif key == 'batchnorm':
            return nn.BatchNorm1d(*args)
        elif key == 'flatten':
            return nn.Flatten()
        elif key == 'relu':
            return nn.ReLU()
        elif key == 'OTHERS':
            pass

    def init_hidden_states(self):
        if self.LSTM_PARAMS[6]:
            NUM_OF_DIRECTIONS = 2
        else:
            NUM_OF_DIRECTIONS = 1
        
        self.hidden = (torch.randn(self.LSTM_PARAMS[2] * NUM_OF_DIRECTIONS, self.DICT['OTHERS']['1']['batchsize'], self.LSTM_PARAMS[1]).to(device=cuda),
                  torch.randn(self.LSTM_PARAMS[2] * NUM_OF_DIRECTIONS, self.DICT['OTHERS']['1']['batchsize'], self.LSTM_PARAMS[1]).to(device=cuda))



    def forward(self,x):
        for layer in self.layers:
            if isinstance(layer, nn.LSTM):

                batchsize , features , windowlength = x.shape

                x = x.reshape(batchsize, windowlength, features)
                x, self.hidden = layer(x,self.hidden)
                batchsize, windowlength, features = x.shape
                x = x.reshape(batchsize, features, windowlength)
            else:
                x = layer(x)
        return x
    
    
    def loss_batch(self, TR_INP, TR_OUT, opt=None):
        loss = self.Loss_FUNC(self(TR_INP), TR_OUT)
        if opt is not None:
            loss.backward(retain_graph=True)
            opt.step()
            opt.zero_grad()
        return loss.item(), len(TR_INP)

    def fit(self):
        self.hist = list()
        self.hist_valid = list()
        min_loss = 10
        BEST_LOSS = 5
        BEST_CHECKPOINT_LOSS = 5
        for epoch in range(self.epoch):
            self.train()
            batch_loss = 0
            count = 0
            print(epoch)
            
            for TR_INP, TR_OUT in self.train_DL:
                losses, nums = self.loss_batch(TR_INP, TR_OUT,opt = self.optimizer)
                batch_loss = batch_loss + losses
                count = count + 1
            train_loss = batch_loss / count
            self.hist.append(train_loss)
            

            self.eval()
            with torch.no_grad():
                losses, nums = zip(
                    *[self.loss_batch(TR_INP, TR_OUT) for TR_INP, TR_OUT in self.val_DL]
                )
            val_loss = np.sum(np.multiply(losses, nums)) / np.sum(nums)
            print('Train Loss:  {}  and Validation Loss:  {}'.format(train_loss, val_loss))

            self.hist_valid.append(val_loss)
            if epoch % 20 == 0 and epoch != 0:
                is_best = val_loss < BEST_CHECKPOINT_LOSS
                BEST_CHECKPOINT_LOSS = min(val_loss,BEST_CHECKPOINT_LOSS)
                self.save_checkpoint({
                                  'epoch': epoch + 1,
                                  'state_dict': self.state_dict(),
                                  'BEST_CHECKPOINT_LOSS': BEST_CHECKPOINT_LOSS,
                                  'optimizer' : self.optimizer.state_dict(),
                                }, is_best,epoch)

            is_best = val_loss < BEST_LOSS
            BEST_LOSS = min(val_loss,BEST_LOSS)
            print('{}:   {}'.format(epoch,val_loss))

        return BEST_LOSS
      
    def save_checkpoint(self, state, is_best, epoch):
        dirr = self.TRIAL_DIR + '/' + str(epoch) + 'pth.tar'
        best_dirr = self.TRIAL_DIR + '/BEST_IS_' + str(epoch) + 'pth.tar'

        torch.save(state, dirr)
        if is_best:
            torch.save(state, best_dirr)



def SET_EXPERIMENT(PARAMS_TO_CHANGE=None):
    global P_OBJ
    P_OBJ = PARAMETERS()

    P_OBJ.EXPERIMENT_NUMBER = 1
    P_OBJ.GET_DICT()
    P_OBJ.GET_PARAMS_TO_CHANGE()
    P_OBJ.CREATE_SEARCH_SPACE()
    P_OBJ.CREATE_DIR()

    P_OBJ.WRITE_CONSTANTS()

    P_OBJ.preprocess()
    
    best = fmin(fn=P_OBJ.GET_MODEL,
            space=P_OBJ.space,
            algo=tpe.suggest,
            max_evals=50)
    

def SINGLE_RUN():
    OTHERS  =  {
                    'windowlength': 48,
                    'out_size': 4,
                    'period': 24,
                    'lrate': 0.0008,
                    'batchsize': 16,
                    'epoch': 250
                    }


    DICT =  {'CONV': {
                      '1': {'FIL': 128, 
                            'KER': 8,
                            'stride': 1,
                            'padding': 0,
                            'dilation': 2,
                            'dropout': [True, 0.6],
                            'batchnorm': True,
                            'activation_function': [True, 'relu'],
                            'pooling': [False, 0, None]
                          },
                      '2': {'FIL': 96, 
                            'KER': 4,
                            'stride': 1,
                            'padding': 0,
                            'dilation': 2,
                            'dropout': [True, 0.6],
                            'batchnorm': True,
                            'activation_function': [True, 'relu'],
                            'pooling': [False, 2, None]
                          }
                    },

          'flatten': {'1': {'nofilter':0 , 'nonothing':0 }},

          'DENSE': {
          
                    '1': {'FIL': 100,
                          'dropout' : [True,0.6],
                          'activation_function': [True, 'relu']
                        },

                    '2': {'FIL':OTHERS['out_size'],
                          'dropout' : [False,0],
                          'activation_function': [False, '-']
                          }
                  },
            
          'OTHERS': {'1':OTHERS}
              }


    global P_OBJ

    P_OBJ = PARAMETERS()
    P_OBJ.EXPERIMENT_NUMBER = 1
    P_OBJ.GET_DICT(DICT)
    P_OBJ.GET_PARAMS_TO_CHANGE({'OTHERS':{'1':{'lrate':(0.001,0.0001)}}})
    P_OBJ.CREATE_SEARCH_SPACE()
    P_OBJ.CREATE_DIR()

    P_OBJ.WRITE_CONSTANTS()

    P_OBJ.preprocess()

    change = {'OTHERS':{'1':{'lrate':0.003}}}
    P_OBJ.GET_MODEL(change)
    """for x in range(1,10):
        lrate = 0.0002*x
        change = {'OTHERS':{'1':{'lrate':lrate}}}
        P_OBJ.GET_MODEL(change)"""


