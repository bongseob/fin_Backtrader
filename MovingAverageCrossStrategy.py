
import backtrader as bt
import yfinance as yf
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt

class MovingAverageCrossStrategy(bt.Strategy):
    """
    이동평균 교차 전략
    - 단기 이동평균이 장기 이동평균을 상향 돌파하면 매수
    - 단기 이동평균이 장기 이동평균을 하향 돌파하면 매도
    """
    
    params = (
        ('short_period', 10),    # 단기 이동평균 기간
        ('long_period', 30),     # 장기 이동평균 기간
        ('printlog', True),      # 거래 로그 출력 여부
    )
    
    def __init__(self):
        # 종가 데이터 참조
        self.dataclose = self.datas[0].close
        
        # 이동평균 계산
        self.sma_short = bt.indicators.SimpleMovingAverage(
            self.datas[0], period=self.params.short_period)
        self.sma_long = bt.indicators.SimpleMovingAverage(
            self.datas[0], period=self.params.long_period)
        
        # 교차 신호 생성
        self.crossover = bt.indicators.CrossOver(self.sma_short, self.sma_long)
        
        # 거래 카운터
        self.order = None
        self.buyprice = None
        self.buycomm = None
        
    def notify_order(self, order):
        """주문 상태 알림"""
        if order.status in [order.Submitted, order.Accepted]:
            return
        
        if order.status in [order.Completed]:
            if order.isbuy():
                self.log(f'매수 체결: 가격 {order.executed.price:.2f}, '
                        f'수량 {order.executed.size}, '
                        f'수수료 {order.executed.comm:.2f}')
                self.buyprice = order.executed.price
                self.buycomm = order.executed.comm
            else:
                self.log(f'매도 체결: 가격 {order.executed.price:.2f}, '
                        f'수량 {order.executed.size}, '
                        f'수수료 {order.executed.comm:.2f}')
                
        elif order.status in [order.Canceled, order.Margin, order.Rejected]:
            self.log('주문 취소/거부')
            
        self.order = None
        
    def notify_trade(self, trade):
        """거래 완료 알림"""
        if not trade.isclosed:
            return
            
        self.log(f'거래 완료: 수익 {trade.pnl:.2f}, 순수익 {trade.pnlcomm:.2f}')
        
    def log(self, txt, dt=None):
        """로그 출력"""
        if self.params.printlog:
            dt = dt or self.datas[0].datetime.date(0)
            print(f'{dt.isoformat()}: {txt}')
            
    def next(self):
        """매 봉마다 실행되는 메인 로직"""
        # 현재 포지션 확인
        if not self.position:
            # 포지션이 없을 때 - 매수 신호 확인
            if self.crossover > 0:  # 단기선이 장기선을 상향 돌파
                self.log(f'매수 신호: 현재가 {self.dataclose[0]:.2f}')
                # 전체 자금의 95%로 매수
                size = int(self.broker.getcash() * 0.95 / self.dataclose[0])
                self.order = self.buy(size=size)
                
        else:
            # 포지션이 있을 때 - 매도 신호 확인
            if self.crossover < 0:  # 단기선이 장기선을 하향 돌파
                self.log(f'매도 신호: 현재가 {self.dataclose[0]:.2f}')
                self.order = self.sell(size=self.position.size)

def run_backtest():
    """백테스팅 실행 함수"""
    
    # 1. Cerebro 엔진 생성
    cerebro = bt.Cerebro()
    
    # 2. 전략 추가
    cerebro.addstrategy(MovingAverageCrossStrategy)
    
    # 3. 데이터 다운로드 (Apple 주식, 2020-2023)
    print("데이터 다운로드 중...")
    data = yf.download('AAPL', start='2020-01-01', end='2023-12-31')
    
    # 4. Backtrader 데이터 피드 생성
    data_feed = bt.feeds.PandasData(
        dataname=data,
        datetime=None,  # 인덱스를 datetime으로 사용
        open='Open',
        high='High', 
        low='Low',
        close='Close',
        volume='Volume',
        openinterest=None
    )
    
    # 5. 데이터 피드 추가
    cerebro.adddata(data_feed)
    
    # 6. 초기 자금 설정
    cerebro.broker.setcash(100000.0)
    
    # 7. 수수료 설정 (0.1%)
    cerebro.broker.setcommission(commission=0.001)
    
    # 8. 분석기 추가
    cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name='sharpe')
    cerebro.addanalyzer(bt.analyzers.DrawDown, _name='drawdown')
    cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name='trades')
    cerebro.addanalyzer(bt.analyzers.Returns, _name='returns')
    
    # 9. 백테스팅 시작 전 자금 출력
    print(f'백테스팅 시작 자금: {cerebro.broker.getvalue():.2f}')
    
    # 10. 백테스팅 실행
    results = cerebro.run()
    strat = results[0]
    
    # 11. 최종 자금 출력
    print(f'백테스팅 완료 자금: {cerebro.broker.getvalue():.2f}')
    
    # 12. 성과 분석 결과 출력
    print('\n=== 백테스팅 결과 ===')
    
    # 샤프 비율
    sharpe = strat.analyzers.sharpe.get_analysis()
    print(f'샤프 비율: {sharpe.get("sharperatio", "N/A"):.4f}')
    
    # 최대 낙폭
    drawdown = strat.analyzers.drawdown.get_analysis()
    print(f'최대 낙폭: {drawdown.get("max", {}).get("drawdown", "N/A"):.2f}%')
    
    # 거래 분석
    trades = strat.analyzers.trades.get_analysis()
    total_trades = trades.get('total', {}).get('total', 0)
    won_trades = trades.get('won', {}).get('total', 0)
    lost_trades = trades.get('lost', {}).get('total', 0)
    
    print(f'총 거래 횟수: {total_trades}')
    print(f'수익 거래: {won_trades}')
    print(f'손실 거래: {lost_trades}')
    
    if total_trades > 0:
        win_rate = (won_trades / total_trades) * 100
        print(f'승률: {win_rate:.2f}%')
    
    # 수익률
    returns = strat.analyzers.returns.get_analysis()
    total_return = returns.get('rtot', 0) * 100
    print(f'총 수익률: {total_return:.2f}%')
    
    # 13. 차트 출력
    print('\n차트를 생성하고 있습니다...')
    cerebro.plot(style='candlestick', barup='green', bardown='red')
    plt.show()
    
    return cerebro, results

if __name__ == '__main__':
    # 백테스팅 실행
    cerebro, results = run_backtest()
    
    print('\n백테스팅이 완료되었습니다!')
    print('차트에서 다음을 확인할 수 있습니다:')
    print('- 가격 차트 (캔들스틱)')
    print('- 단기/장기 이동평균선')
    print('- 매수/매도 신호점')
    print('- 포트폴리오 가치 변화')

