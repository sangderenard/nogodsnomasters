def kernel(feed0, feed1, feed2):
    v0 = feed0 * feed2
    v1 = 1.0 - feed2
    v2 = feed1 * v1
    v3 = v0 + v2
    v4 = v3 * 255.0
    return v4
