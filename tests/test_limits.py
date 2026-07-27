from src.risk.limits import RiskLimits


def _rl():
    return RiskLimits(max_gamma=500.0, max_vega=10000.0, max_contracts_per_leg=50)


def test_full_size_when_flat():
    bid, ask = _rl().adjusted_quote_sizes(
        desired_size=10, portfolio_gamma=0.0, portfolio_vega=0.0, current_leg_position=0
    )
    assert (bid, ask) == (10, 10), "a flat book has full headroom on both sides"


def test_greek_throttle_hits_only_the_increasing_side():
    # Long gamma and long vega at 80% of cap: buying adds to both, selling reduces.
    bid, ask = _rl().adjusted_quote_sizes(
        desired_size=10, portfolio_gamma=400.0, portfolio_vega=8000.0, current_leg_position=0
    )
    assert bid == 2, "buy side throttled to the 20% remaining headroom"
    assert ask == 10, "sell side reduces exposure and must not be throttled"


def test_greek_throttle_reverses_when_book_is_short():
    # Short gamma and short vega: now SELLING adds exposure and buying reduces it.
    bid, ask = _rl().adjusted_quote_sizes(
        desired_size=10, portfolio_gamma=-400.0, portfolio_vega=-8000.0, current_leg_position=0
    )
    assert bid == 10
    assert ask == 2


def test_leg_cap_blocks_only_the_increasing_side():
    bid, ask = _rl().adjusted_quote_sizes(
        desired_size=10, portfolio_gamma=0.0, portfolio_vega=0.0, current_leg_position=47
    )
    assert bid == 3, "only 3 contracts of room left to go longer"
    assert ask == 10, "reducing a long position is unconstrained by the long cap"


def test_book_at_the_cap_can_still_quote_its_way_out():
    """The defect this replaced: at the cap the old throttle refused BOTH sides.

    A book pinned at +50 could not sell, so it could only exit via expiry.
    """
    bid, ask = _rl().adjusted_quote_sizes(
        desired_size=10, portfolio_gamma=0.0, portfolio_vega=0.0, current_leg_position=50
    )
    assert bid == 0, "no room to add"
    assert ask == 10, "must still be able to reduce"

    bid, ask = _rl().adjusted_quote_sizes(
        desired_size=10, portfolio_gamma=0.0, portfolio_vega=0.0, current_leg_position=-50
    )
    assert bid == 10
    assert ask == 0


def test_sizes_never_negative():
    bid, ask = _rl().adjusted_quote_sizes(
        desired_size=10, portfolio_gamma=9999.0, portfolio_vega=999999.0,
        current_leg_position=999,
    )
    assert bid >= 0 and ask >= 0
