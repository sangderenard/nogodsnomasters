def kernel(feed0):
    v0 = feed0.cos()
    v1 = 0.5 * v0
    v2 = 0.5 + v1
    v3 = feed0 - 2.094395102393195
    v4 = v3.cos()
    v5 = 0.5 * v4
    v6 = 0.5 + v5
    v7 = feed0 - 4.18879020478639
    v8 = v7.cos()
    v9 = 0.5 * v8
    v10 = 0.5 + v9
    return (feed0, v2, v6, v10)
