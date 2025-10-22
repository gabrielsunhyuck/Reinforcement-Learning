import torch
import torch.nn            as nn
import torch.nn.functional as F
import torch.optim         as optim
import numpy               as np
import matplotlib.pyplot   as plt

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        '''
        (1) nn.Linear : 하나의 히든 레이어를 의미한다.
        (2) nn.Linear 내부 변수는 앞의 레이어 노드 개수 / 뒤의 레이어 노드 개수를 의미한다.
        --> nn.Linear(1, 128) : 1*128개의 파라미터(w)가 필요하다는 것으 의미한다.
        --> 현재 구조는 인풋이 거쳐가는 히든 레이어 개수가 4개로 설정한다는 의미이다.
        '''
        self.f1 = nn.Linear(1  , 128)
        self.f2 = nn.Linear(128, 128)
        self.f3 = nn.Linear(128, 128)
        self.f4 = nn.Linear(128,   1, bias=False)

    def forward(self, x):
        '''
        히든 레이어 4개를 갖춘 모델을 이용하여 실제로 연산할 떄 호출되는 함수
        비선형 함수인 "relu"가 포함되어 있다.
        '''
        x = F.relu(self.f1(x))
        x = F.relu(self.f2(x))
        x = F.relu(self.f3(x))
        x = self.f4(x)
        return x
    
def data_processing(x):
    '''
    데이터를 생성하는 함수 (numpy) :: 실제 함수
    (1) 이 함수를 통해 나오는 F(x)의 값과 실제 데이터를 통해 손실 함수를 계산
    (2) 손실 함수를 바탕으로 그라디언트 계산
    '''
    noise = np.random.rand(x.shape[0]) * 0.4 - 0.2
    return np.cos(1.5 * np.pi * x) + x + noise
    
def plotting(model):
    '''
    실제 데이터와 학습시킨 모델을 이요한 예측치를 그래프로 그리는 함수
    '''
    # (1) 0~5사이 100개의 숫자를 등간격으로 뽑아서 인풋(in_x)으로 생성
    x    = np.linspace(0, 5, 100)
    in_x = torch.from_numpy(x).float().unsqueeze(1)
    plt.plot(x, data_processing(x), label="Truth")
    plt.plot(x, model(in_x).detach().numpy(), label="Prediction")
    plt.legend(loc='lower right', fontsize=15)
    plt.xlim(( 0, 5))
    plt.ylim((-1, 5))
    plt.grid()
    plt.show()