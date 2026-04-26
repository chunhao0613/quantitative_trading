from __future__ import annotations

import backtrader as bt


class TaiwanStockCommission(bt.CommInfoBase):
    params = (
        ("fee_rate", 0.001425),
        ("tax_rate", 0.003),
    )

    def _getcommission(self, size, price, pseudoexec):
        fee = abs(size) * price * self.p.fee_rate
        tax = abs(size) * price * self.p.tax_rate if size < 0 else 0.0
        return fee + tax


class SignalPandasData(bt.feeds.PandasData):
    lines = ("signal", "atr_14")
    params = (
        ("datetime", None),
        ("open", "open"),
        ("high", "high"),
        ("low", "low"),
        ("close", "close"),
        ("volume", "volume"),
        ("openinterest", -1),
        ("signal", "signal"),
        ("atr_14", "atr_14"),
    )


class MLSignalStrategy(bt.Strategy):
    params = (
        ("slip_atr_mult", 0.2),
        ("min_slip", 0.0002),
        ("max_slip", 0.01),
        ("stake", 1000),
    )

    def __init__(self):
        self.order = None

    def next(self):
        if self.order:
            return

        close = self.data.close[0]
        atr = self.data.atr_14[0] if self.data.atr_14[0] == self.data.atr_14[0] else 0.0
        dynamic_slip = min(self.p.max_slip, max(self.p.min_slip, (atr / close) * self.p.slip_atr_mult if close > 0 else self.p.min_slip))

        self.broker.set_slippage_perc(
            perc=dynamic_slip,
            slip_open=True,
            slip_limit=True,
            slip_match=True,
            slip_out=False,
        )

        if not self.position and self.data.signal[0] > 0:
            self.order = self.buy(size=self.p.stake)
        elif self.position and self.data.signal[0] <= 0:
            self.order = self.sell(size=self.p.stake)

    def notify_order(self, order):
        if order.status in [order.Completed, order.Canceled, order.Margin, order.Rejected]:
            self.order = None
