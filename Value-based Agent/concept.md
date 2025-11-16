# 가치 기반 에이전트 (Value-based Agent)

## [ 학습 방향 ]   
- 어떠한 제약 조건도 없는 상황
- 모델 프리 상황
- 상태 공간(State space)과 액션 공간(Action space)이 매우 커서 밸류를 일일이 테이블에 담지 못하는 상황에서의 해결책 필요
- 뉴럴넷과 강화학습의 접목
> **강화 학습에 뉴럴넷을 접목시키는 방법에는 두 가지가 있음**
> - 가치 함수를 뉴럴넷으로 표현하는 방식
> - 정책 함수 자체를 뉴럴넷으로 표현하는 방식

#### (1) 가치 기반 에이전트 (Value-based)
- 가치 기반 에이전트는 가치 함수에 근거하여 액션을 선택한다.  
- 액션-가치 함수 $q(s,a)$의 값을 보고 액션을 선택하는 것이다.  
- 상태 $s$에서 선택할 수 있는 액션들 중에서 가장 밸류가 높은 액션을 선택하는 방식
- EX) Q-러닝, SARSA 등

#### (2) 정책 기반 에이전트 (Policy-based)
- 정책 함수 $\pi(s, a)$를 보고 직접 액션을 선택한다.
- 정책만 있다면 에이전트는 MDP 안에서 경험을 쌓을 수 있고, 이 경험을 이용해 학습 과정에서 정책을 강화한다.
- 가치 함수를 활용하지 않는다.

#### (3) 액터-크리틱 (Actor-Critic)
- 가치 함수와 정책 함수 모두를 사용한다.
- 액터 : 정책 $(\pi)$
- 크리틱 : 가치 함수 $(v(s), q(s,a))$
  


## 밸류 네트워크의 학습
### 밸류 네트워크 
- 뉴럴넷으로 이루어진 가치 함수 $v_{\theta,\pi}(s)$
- $\theta$ : 뉴럴넷의 파라미터 (초기에는 랜덤으로 초기화)
- 적절한 $\theta$를 학습하여 가치 함수가 각 상태별로 올바른 밸류를 출력하도록 하는 것이 목표이다.
- 손실 함수 : 예측과 정답 사이의 차이
$$L(\theta)=\mathbb{E}_\pi[(v_{true}(s)-v_{\theta}(s))^2]$$
$$\nabla_\theta L(\theta)=-\mathbb{E}_\pi [(v_{true}(s)-v_{\theta}(s))^2]$$
>기댓값을 계산하기 위해서는 정책을 이용하여 움직이는 에이전트를 통해 샘플을 뽑아야 한다.
$$\theta'=\theta-\alpha\nabla_\theta L(\theta)=\theta + \alpha(v_{true}(s)-v_\theta(s))\nabla_\theta v_\theta (s)$$
> $\alpha$ : 파라미터 $\theta$를 얼만큼 업데이트 할 것인지에 대한 결정

 실제 가치 함수를 모른다면 (실제로 주어질 일은 만무함) 정답이 주어지지 않은 것이고, 정답지가 없다면 손실 함수도 정의가 불가능하며 그라디언트 계산도 불가능해짐. 따라서, **몬테카를로 방법을 사용한 리턴**과 **TD 학습 방법**을 사용할 수 있다.

#### (1) 몬테카를로 리턴
$$L(\theta)=\mathbb{E}_\pi[(G_t-v_{\theta}(s))^2]$$
$$\nabla_\theta L(\theta)=-\mathbb{E}_\pi [(G_t-v_{\theta}(s))^2]$$
$$\theta'=\theta-\alpha\nabla_\theta L(\theta)=\theta + \alpha(G_t-v_\theta(s))\nabla_\theta v_\theta (s)$$
- $G_t$ : 에피소드가 끝날 때까지 얻은 감쇠된 누적 보상
#### (2) TD 타깃
- TD 한습 방법은 한 스텝 더 진행해서 추측한 값을 이용하여 현재의 추측값을 업데이트하는 방식
- 누적 보상 대신에 TD 타깃인 $r_{t+1}+\gamma v_\theta(s_{t+1})$를 대입한다.
$$L(\theta)=\mathbb{E}_\pi[(r_{t+1}+\gamma v_\theta(s_{t+1})-_{\theta}(s))^2]$$
$$\theta'=\theta-\alpha\nabla_\theta L(\theta)=\theta + \alpha(r_{t+1}+\gamma v_\theta(s_{t+1})-v_\theta(s))\nabla_\theta v_\theta (s)$$
- $r_{t+1}+\gamma v_\theta(s_{t+1})$값은 변수가 아닌 상수이다.

## 딥 Q러닝