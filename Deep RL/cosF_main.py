from cosF_model import *

def main():
    # (1) 0~5 사이의 숫자 1만개를 샘플링하여 입력으로 사용
    data_x    = np.random.rand(10000)*5
    # cosF_model.py에서 Model 클래스 호출
    model     = Model()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    for step in range(10000):
        # (2) 뉴럴넷 학습은 미니 배치 단위로 이루어짐
        # --> 1만 개의 데이터 중 랜덤하게 32개를 뽑아서 미니 배치로 재구성
        batch_x = np.random.choice(data_x, 32)
        # (3) 파이토치의 텐서로 변환하여 모델에 입력
        batch_x_tensor = torch.from_numpy(batch_x).float().unsqueeze(1)
        pred = model(batch_x_tensor)

        # (4) 미니 배치를 통해 샘플링한 데이터를 바탕으로 데이터 생성 함수의 결과를 산출 (실제 값)
        batch_y = data_processing(batch_x)
        # (5) 파이토치의 텐서로 변환
        truth   = torch.from_numpy(batch_y).float().unsqueeze(1)
        # (6) 손실 함수 (truth - pred)
        loss    = F.mse_loss(pred, truth)

        optimizer.zero_grad()
        # 역전파 알고리즘을 통해 그라디언트 계산을 수행
        loss.mean().backward()
        # 파라미터(w) 업데이트
        optimizer.step()
    
    plotting(model)
