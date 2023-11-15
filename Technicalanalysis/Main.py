
class Order:
    def __init__(self, timestamp, bought_at, stop_loss, take_profit, order_type, sold_at=None,
                 is_active: bool = True):
        self.timestmap = timestamp
        self.bought_at = bought_at
        self.sold_at = sold_at
        self.stop_loss = stop_loss
        self.take_profit = take_profit
        self.order_type = order_type
        self.is_active = is_active

    def __repr__(self) -> str:
        return f"{self.order_type} Position - id {self.timestamp}"

    def __str__(self) -> str:
        return f"{self.order_type} {self.is_active} Position @ {self.bought_at}"



class Order2:
    def __init__(self2, timestamp2, bought_at2, stop_loss2, take_profit2, order_type2, sold_at2=None,
                 is_active2: bool = True):
        self2.timestmap2 = timestamp2
        self2.bought_at2 = bought_at2
        self2.sold_at2 = sold_at2
        self2.stop_loss2 = stop_loss2
        self2.take_profit2 = take_profit2
        self2.order_type2 = order_type2
        self2.is_active2 = is_active2

    def __repr__(self2) -> str:
        return f"{self2.order_type2} Position - id {self.timestamp2}"

    def __str__(self2) -> str:
        return f"{self.order_type2} {self.is_active2} Position @ {self.bought_at2}"


