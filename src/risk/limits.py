class RiskLimits:
    def __init__(self, max_gamma: float, max_vega: float, max_contracts_per_leg: int):
        self.max_gamma             = max_gamma
        self.max_vega              = max_vega
        self.max_contracts_per_leg = max_contracts_per_leg

    def _greek_size(self, desired_size: int, exposure: float, cap: float,
                    side_increases: bool) -> int:
        """Throttle only the side that INCREASES an exposure toward its cap.

        The reducing side is never throttled: a maker that cannot quote its way back
        toward flat is not managing risk, it is trapped.
        """
        if not side_increases:
            return desired_size
        headroom = max(0.0, cap - abs(exposure))
        return max(0, int(desired_size * (headroom / cap)))

    def adjusted_quote_sizes(self, desired_size: int, portfolio_gamma: float,
                             portfolio_vega: float, current_leg_position: int):
        """Return (bid_size, ask_size) — the size shown on each side of the quote.

        The MM buys on its bid and sells on its ask, so for a leg held long the ask is
        the inventory-reducing side and the bid is the increasing one. Throttling on
        abs(position) refused BOTH sides at the cap, so a book at +20 per leg froze and
        could only exit via expiry.

        The portfolio Greek caps are directional for the same reason. Which side adds
        exposure depends on the sign the book already carries: a long-vega book adds vega
        by buying, a short-vega book adds it by selling. When the book is flat on a Greek,
        headroom is full and neither side is throttled.
        """
        # Per-leg headroom, signed: room to go longer vs room to go shorter.
        buy_headroom  = max(0, self.max_contracts_per_leg - current_leg_position)
        sell_headroom = max(0, self.max_contracts_per_leg + current_leg_position)

        # Buying options adds positive gamma/vega, selling adds negative. So the buy side
        # increases |exposure| exactly when the book already carries that Greek long.
        buy_adds_gamma = portfolio_gamma >= 0
        buy_adds_vega  = portfolio_vega  >= 0

        bid_size = min(
            buy_headroom,
            self._greek_size(desired_size, portfolio_gamma, self.max_gamma, buy_adds_gamma),
            self._greek_size(desired_size, portfolio_vega,  self.max_vega,  buy_adds_vega),
        )
        ask_size = min(
            sell_headroom,
            self._greek_size(desired_size, portfolio_gamma, self.max_gamma, not buy_adds_gamma),
            self._greek_size(desired_size, portfolio_vega,  self.max_vega,  not buy_adds_vega),
        )
        return max(0, bid_size), max(0, ask_size)
