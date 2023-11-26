# pattern of wally shirt
# pattern + 1, +1, -1, -1 on vertical axis
pattern = np.ones((24, 16), float)
for i in range(2):
    pattern[i::4] = -1