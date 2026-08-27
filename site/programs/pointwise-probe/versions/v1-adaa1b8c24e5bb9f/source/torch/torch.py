def kernel(feed0):
    v0 = torch.cos(feed0)
    v1 = 0.5 * v0
    v2 = 0.5 + v1
    v3 = feed0 - 2.094395102393195
    v4 = torch.cos(v3)
    v5 = 0.5 * v4
    v6 = 0.5 + v5
    v7 = feed0 - 4.18879020478639
    v8 = torch.cos(v7)
    v9 = 0.5 * v8
    v10 = 0.5 + v9
    return (feed0, v2, v6, v10)
