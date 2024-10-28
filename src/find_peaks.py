import numpy as np

from src.random_forest import load_curve
from matplotlib import pyplot as plt


def findpeaksSG(x, y, SlopeThreshold, AmpThreshold, smoothwidth, peakgroup, smoothtype=1):
    if smoothtype > 3:
        smoothtype = 3
    if smoothtype < 1:
        smoothtype = 1
    if smoothwidth < 1:
        smoothwidth = 1
    if np.isscalar(AmpThreshold):
        AmpThreshold = AmpThreshold * np.ones_like(SlopeThreshold)
    if np.isscalar(smoothwidth):
        smoothwidth = smoothwidth * np.ones_like(SlopeThreshold)
    if np.isscalar(peakgroup):
        peakgroup = peakgroup * np.ones_like(SlopeThreshold)

    smoothwidth = np.round(smoothwidth).astype(int)
    peakgroup = np.round(peakgroup).astype(int)

    if smoothwidth > 1:
        d = SegmentedSmooth(deriv(y), smoothwidth, smoothtype)
    else:
        d = deriv(y)

    P = np.zeros((0, 5))
    vectorlength = len(y)
    NumSegs = len(SlopeThreshold)
    peak = 1

    # for j in range(2 * round(smoothwidth[0] / 2) - 1, len(y) - smoothwidth[0] - 1):
    for j in range(2 * round(smoothwidth[0] / 2) - 2, len(y) - smoothwidth[0] - 1):  #
        Seg = int(1 + NumSegs / (vectorlength / j)) - 1  #
        n = round(peakgroup[Seg] / 2 + 1)

        if np.sign(d[j]) > np.sign(d[j + 1]):  # Detects zero-crossing
            if d[j] - d[j + 1] > SlopeThreshold[Seg]:  # if slope of derivative is larger than SlopeThreshold
                if y[j] > AmpThreshold[Seg]:  # if height of peak is larger than AmpThreshold
                    xx = np.zeros(peakgroup[Seg])
                    yy = np.zeros(peakgroup[Seg])
                    for k in range(peakgroup[Seg]):
                        groupindex = j + k - n + 2
                        groupindex = max(1, min(groupindex, vectorlength))
                        xx[k] = x[groupindex - 1]  # Adjust for 0-based index
                        yy[k] = y[groupindex - 1]  # Adjust for 0-based index

                    if peakgroup[Seg] > 2:
                        Height, Position, Width = gaussfit(xx, yy)
                        PeakX = np.real(Position)  # Compute peak position and height of fitted parabola
                        PeakY = np.real(Height)
                        MeasuredWidth = np.real(Width)
                    else:
                        PeakY = np.max(yy)
                        pindex = np.where(yy == PeakY)[0]
                        PeakX = xx[pindex[0]]
                        MeasuredWidth = 0

                    if np.isnan(PeakX) or np.isnan(PeakY) or PeakY < AmpThreshold[Seg]:
                        continue  # Skip this peak
                    else:  # Otherwise count this as a valid peak
                        P = np.vstack([P, [round(peak), PeakX, PeakY, MeasuredWidth, 1.0646 * PeakY * MeasuredWidth]])
                        peak += 1  # Move on to next peak
    return P


def SegmentedSmooth(y, smoothwidths, type=1, ends=0):
    ly = len(y)
    NumSegments = len(smoothwidths)
    SegLength = round(ly / NumSegments)
    SmoothSegment = np.zeros((ly, NumSegments))
    SmoothedSignal = np.zeros(ly)

    for Segment in range(NumSegments):
        SmoothSegment[:, Segment] = fastsmooth(y, smoothwidths[Segment], type, ends)
        startindex = 1 + Segment * SegLength
        endindix = startindex + SegLength - 1
        if endindix > ly:
            endindix = ly
        indexrange = range(startindex - 1, endindix)  # Adjust for 0-based indexing
        SmoothedSignal[indexrange] = SmoothSegment[indexrange, Segment]

    return SmoothedSignal


def deriv(a):
    # First derivative of vector using 2-point central difference.
    n = len(a)
    d = np.zeros_like(a)
    d[0] = a[1] - a[0]
    d[-1] = a[-1] - a[-2]
    for j in range(1, n-1):
        d[j] = (a[j+1] - a[j-1]) / 2
    return d


def gaussfit(x, y):
    maxy = np.max(y)
    y = np.where(y < (maxy / 100), maxy / 100, y)

    logyyy = np.log(np.abs(y))
    coef = np.polyfit(x, logyyy, 2)
    c1, c2, c3 = coef[2], coef[1], coef[0]

    # Compute peak position and height of fitted parabola
    Position = -((coef[1] * 2 / (2 * c3)) - np.mean(x))
    Height = np.exp(c1 - c3 * (c2 / (2 * c3)) ** 2)
    Width = np.linalg.norm(2.35703 * 2 / (np.sqrt(2) * np.sqrt(-1 * c3)))

    return Height, Position, Width


def fastsmooth(Y, w, type, ends):
    if type == 1:
        SmoothY = sa(Y, w, ends)
    elif type == 2:
        SmoothY = sa(sa(Y, w, ends), w, ends)
    elif type == 3:
        SmoothY = sa(sa(sa(Y, w, ends), w, ends), w, ends)
    elif type == 4:
        SmoothY = sa(sa(sa(sa(Y, w, ends), w, ends), w, ends), w, ends)
    elif type == 5:
        SmoothY = sa(sa(sa(sa(Y, round(1.6 * w), ends), round(1.4 * w), ends), round(1.2 * w), ends), w, ends)
    return SmoothY


def sa(Y, smoothwidth, ends):
    w = round(smoothwidth)
    SumPoints = sum(Y[:w])
    s = np.zeros_like(Y)
    halfw = round(w / 2)
    L = len(Y)
    for k in range(L - w):
        s[k + halfw - 1] = SumPoints
        SumPoints = SumPoints - Y[k]
        SumPoints = SumPoints + Y[k + w]
    s[k + halfw] = sum(Y[-w:])
    SmoothY = s / w
    if ends == 1:
        startpoint = (smoothwidth + 1) // 2
        SmoothY[0] = (Y[0] + Y[1]) / 2
        for k in range(1, startpoint):
            SmoothY[k] = np.mean(Y[:2 * k])
            SmoothY[-k] = np.mean(Y[-2 * k:])
        SmoothY[-1] = (Y[-1] + Y[-2]) / 2
    return SmoothY


if __name__ == '__main__':
    def gaussian_function(x, mu, sigma):
        y = (1 / (sigma * np.sqrt(2 * np.pi))) * np.exp(-((x - mu) ** 2) / (2 * sigma ** 2))
        return y
    # Y, label = load_curve(r'../data/dataset/exports/3.txt')
    # X = np.arange(0, 1 + 1/(len(Y)-1), 1/(len(Y)-1))

    X = np.arange(1, 100.2, 0.2)
    Y = gaussian_function(X, 20, 1.5) + gaussian_function(X, 80, 30) + 0.02 * np.random.randn(len(X))
    # res = findpeaksSG(X, Y, [0.001], .2, 5, 10, 3)
    res = findpeaksSG(X, Y, [0.0001], .2, 5, 10, 3)
    print(res)

    plt.plot(X, Y)
    plt.show()


