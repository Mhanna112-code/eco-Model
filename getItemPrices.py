def getItemPrices(text):
    prices = []
    f = open(text, 'r')
    for line in f:
        if line[0] == 0:
            prices.append(6.54)
        if line[0] == 1:
            prices.append(11.70)
        if line[0] == 2:
            prices.append(28.00)
        if line[0] == 3:
            prices.append(28.00)
        if line[0] == 4:
            prices.append(34.42)
        if line[0] == 5:
        if line[0] == 6:
        if line[0] == 7:
        if line[0] == 8:
        if line[0] == 9:
        if line[0] == 10: