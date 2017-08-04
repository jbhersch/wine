import numpy as np
import matplotlib.pyplot as plt
from scipy.fftpack import fft, ifft

class FFS(object):
    '''
    DESCRIPTION:

    ATTRIBUTES:
        - x: x-coordinates (ndarray)
        - y: y-coordinates (ndarray)
        - n: number of data points in padded x,y vectors (int)
        - n0: number of data points in original x,y vectors (int)
        - xa: first point of x (float)
        - xb: last point of padded x (float)
        - h: length of spacing in x (float)
        - L: span of padded x plus h (float)
        - m: number of frequencies in Fourier Series (int)
        - a0: amplitude of zero frequency wave in Fourier Series (int)
        - c: complex vector of Fourier Transform coeffcients (ndarray)
        - a: vector of amplitudes for cosine waves in Fourier Series (ndarray)
        - b: vector of amplitudes for sine waves in Fourier Series (ndarray)
    '''
    def __init__(self, x, y, pad = 0, thresh = 0):
        '''
        INPUT:
            - x: x-coordinates (ndarray)
            - y: y-coordinates (ndarray)
            - pad: number of data points to pad the x, y vectors with (int)
            - thresh: threshold used for noise removal (float)
        OUTPUT: None
        '''
        self.n0 = len(x)
        if thresh > 0 and thresh < 1:
            y = self.remove_noise(y, thresh)
        if pad > 0:
            x, y = self.fourier_pad(x, y, pad)
        self.x = x
        self.y = y
        self.n = len(x) # length of x, y
        self.xa = x[0] # min of x
        self.xb = x[-1] # max of x
        L = x[-1] - x[0]
        self.h = 1.*L/(self.n - 1) # increment size of x
        self.L = L + self.h # span of x + h

        yt = fft(y) # fourier transformation of y
        # only use first half of yt for Fourier Series coeffcients
        self.m = int(np.ceil(self.n/2))

        # define a0
        self.a0 = yt[0].real/self.n
        # set c equal to first half of Fourier Coefficients
        self.c = 2*yt[1:self.m]/self.n
        # a = cosine coefficients
        self.a = self.c.real
        # b = sine coefficients
        self.b = -self.c.imag

    def remove_noise(self, y, thresh):
        '''
        INPUT:
            - y: y-coordinates (ndarray)
            - thresh: threshold used for noise removal (float)
        OUTPUT: None
            - y vector after noise removal (ndarray)
        '''
        # run fft on y
        yt = fft(y)

        # extract fft coefficients
        yt0 = yt[0] # zero frequency coefficient
        yt = yt[1:] # non_zero frequency coefficients
        yt_abs = np.absolute(yt) # amplitudes of non_zero frequency coefficients
        yt_max = np.max(yt_abs) # max amplitude coefficient

        # extract indices where yt_abs < thresh*yt_max
        indices = np.where(yt_abs < thresh*yt_max)[0]

        # set yt[indices] to zero - this eliminates low amplitude waves where
        # noise in signal occurs
        yt[indices] = np.complex(0,0)

        # return inverse fft with low amplitude waves zeroed out
        return ifft(np.hstack((yt0, yt)))

    def fourier_pad(self, x, y, p):
        '''
        INPUT:
            - x: x-coordinates (ndarray)
            - y: y-coordinates (ndarray)
            - p: number of padding points (int)
        OUTPUT: None
            - xx: x-coordinates of original x vector with padding (ndarray)
            - yy: y-coordinates of original y vector with padding (ndarray)
        '''
        n = len(x) # length of x,y
        xa = x[0] # first point in x
        xb = x[-1] # last point in x
        L = xb - xa # span of x
        h = L/(n-1) # increment spacing of x
        m = n + p # number of points in xx
        xc = xa + m*h # end point of xx

        xpad = xb + h*np.arange(1,p+1) # padded x-coordinates to combine with x

        # set values used for splines
        x1, x2, x3, x4 = x[-2], x[-1], xc, xc + h
        y1, y2, y3, y4 = y[-2], y[-1], y[0], y[1]

        # linear interpolations
        s12_1 = ((x2 - xpad)*y1 + (xpad - x1)*y2) / (x2 - x1) # between x1 and x2
        s23_1 = ((x3 - xpad)*y2 + (xpad - x2)*y3) / (x3 - x2) # between x2 and x3
        s34_1 = ((x4 - xpad)*y3 + (xpad - x3)*y4) / (x4 - x3) # between x3 and x4

        # quadratic interpolations
        s13_2 = ((x3 - xpad)*s12_1 + (xpad - x1)*s23_1) / (x3 - x1) # between x1 and x3
        s24_2 = ((x4 - xpad)*s23_1 + (xpad - x2)*s34_1) / (x4 - x2) # between x2 and x4

        # cubic interpolations
        s14_3 = ((x4 - xpad)*s13_2 + (xpad - x1)*s24_2) / (x4 - x1) # between x1 and x4

        # add padding to original x and y vectors
        xx = np.hstack((x, xpad))
        yy = np.hstack((y, s14_3))

        return xx, yy

    def evaluate(self, x, N = None, deriv = 0):
        '''
        INPUT:
            - x: x values where Fourier Series to be evaluated (int, float, or ndarray)
            - N: order of Fourier Series to be evaluated at (int)
            - deriv: derivative of Fourier Series to be evaluated (0, 1, or 2)
        OUTPUT: None
            - ndarray of Fourier Series evaluated at x
        '''
        # if x not an ndarray cast it as one
        if type(x) in [float, int, np.float64]:
            x = np.array([x])
        # initalize y as a0
        # transform x to be in range of xa to xb
        omega = 2*np.pi/self.L
        # x = 2*np.pi*(x - self.xa)/self.L
        x = omega*(x-self.xa)
        # set N, order of Fourier Series
        if N == None:
            N = self.m - 1
        else: # make sure that N is not greater than self.m - 1
            N = np.min([N - 1, self.m - 1])
        # calculate y over order N Fourier Series
        if deriv == 0:
            y = self.a0*np.ones(len(x))
            for k in xrange(0, N):
                y += self.a[k]*np.cos(x*(k+1)) + self.b[k]*np.sin(x*(k+1))
        elif deriv == 1:
            y = np.zeros(len(x))
            for k in xrange(0, N):
                y += (k+1)*(-self.a[k]*np.sin(x*(k+1)) + self.b[k]*np.cos(x*(k+1)))
            y *= omega
        elif deriv == 2:
            y = np.zeros(len(x))
            for k in xrange(0, N):
                y += ((k+1)**2)*(-self.a[k]*np.cos(x*(k+1)) - self.b[k]*np.sin(x*(k+1)))
            y *= omega**2
        return y

def refine_image(img, rows, cols):
    x = np.arange(0, img.shape[1])
    xx = np.linspace(x[0], x[-1], cols)
    y = np.arange(0, img.shape[0])
    yy = np.linspace(x[0], x[-1], rows)

    if len(img.shape) == 3:
        layers = img.shape[2]
    else:
        layers = 1

    img_out = np.zeros((rows,cols,layers))
    for layer in xrange(layers):
        img1 = np.zeros((img.shape[0], cols))
        for n in xrange(img.shape[0]):
            fs = FFS(x, img[n,:,layer], img.shape[1])
            img1[n,:] = fs.evaluate(xx)

        img2 = np.zeros((rows, cols))
        for n in range(cols):
            fs = FFS(y, img1[:,n], img.shape[0])
            img2[:,n] = fs.evaluate(yy)

        img_out[:,:,layer] = img2

    if layers == 1:
        img_out = img_out[:,:,0]

    return img_out

if __name__ == '__main__':
    x = np.linspace(-2,2,50)
    y = np.exp(-x**2)
    #
    # # xx, yy = fourier_pad(x,y,200)
    #
    # # plt.plot(x,y)
    # # plt.show()
    #
    # f = FFS(x,y,0)
    # # xx, yy = f.fourier_pad(x,y,50)
    # ypred = f.evaluate(x)
    # # plt.plot(y, 'r')
    # # plt.plot(ypred, 'b')
    # # plt.show()
    #
    #
    # # xright = np.linspace(6,14,100)
    # plt.plot(x,y,'b', linewidth = 5)
    # plt.plot(x,ypred,'r')
    # # plt.plot(xright, f.eval(xright), 'g')
    # plt.show()

    # y = np.array([1,2,3,4,5])
    # yt = fft(y)
