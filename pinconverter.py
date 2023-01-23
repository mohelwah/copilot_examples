zigzag_len = input(int(9), "ZigZag Length")
show_zigzag = input(True, "Show Zigzag")
fib_factor = input(float( 0.33, "Fib Factor for breakout confirmation", 0, 1, 0.01))


high_points_arr=np.empty(5)
high_index_arr=np.empty(5)
low_points_arr=np.empty(5)
low_index_arr=np.empty(5)
bu_ob_boxes=np.empty(5)
be_ob_boxes=np.empty(5)

to_up = high >= ta.highest(zigzag_len)
to_down = low <= ta.lowest(zigzag_len)



trend = 1
trend = nz(trend[1], 1)





def trend(trend, to_down, to_up):
    if trend == 1 and to_down:
        return -1
    elif trend == -1 and to_up:
        return 1
    else:
        return trend
    

last_trend_up_since = ta.barssince(to_up[1])

