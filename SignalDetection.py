import numpy as np
import scipy.stats as stats
import scipy.optimize as optimize
import matplotlib.pyplot as plt


class SignalDetection:

    def __init__(self, hits, misses, falseAlarms, correctRejections):
        """
        Initializes the class using the self object and the signal detection variables.
        """
        self.hits = hits
        self.misses = misses
        self.falseAlarms = falseAlarms
        self.correctRejections = correctRejections


    def __str__(self):
        """
        Returns the class as a labeled list to enable printing and error detection.
        """
        return f"hits: {self.hits}, misses: {self.misses}, false alarms: {self.falseAlarms}, " \
               f"correct rejections: {self.correctRejections}"


    def hit_rate(self):
        """
        Returns the hit rate based on the class object.
        """
        return self.hits / (self.hits + self.misses)


    def false_alarm_rate(self):
        """
        Returns the false alarm rate based on the class object.
        """
        return self.falseAlarms / (self.falseAlarms + self.correctRejections)


    def d_prime(self):
        """
        Returns the d-prime value given the hit rate and false alarm rate.
        """
        return stats.norm.ppf(self.hit_rate()) - stats.norm.ppf(self.false_alarm_rate())


    def criterion(self):
        """
        Returns the criterion value given the hit rate and false alarm rate.
        """
        return -0.5 * stats.norm.ppf(self.hit_rate())-stats.norm.ppf(self.false_alarm_rate())


    # overloading the + and * methods
    def __add__(self, other):
        """
        add up each type of trial from two objects.
        """
        return SignalDetection(
            self.hits + other.hits,
            self.misses + other.misses,
            self.falseAlarms + other.falseAlarms,
            self.correctRejections + other.correctRejections)


    def __mul__(self, scalar):
        """
        multiply all types of trials with a scalar.
        """
        return SignalDetection(
            self.hits * scalar,
            self.misses * scalar,
            self.falseAlarms * scalar,
            self.correctRejections * scalar)


    @staticmethod
    def simulate(dPrime, criteriaList, signalCount, noiseCount):
        """
        simulate signal detection object
        """
        sdtList = []
        for i in range(len(criteriaList)):
            criteria = criteriaList[i]
            k = criteria + dPrime / 2
            hr = 1 - stats.norm.cdf(k - dPrime)
            far = 1 - stats.norm.cdf(k)
            hits = np.random.binomial(n=signalCount, p=hr)
            misses = signalCount - hits
            falseAlarms = np.random.binomial(n=noiseCount, p=far) #get the hr and far from the dprime
            # and criteria list and then use those to do random number generation for hits, misses,
            # false alarms and correct rejections
            correctRejections = noiseCount - falseAlarms
            sdtList.append(SignalDetection(hits, misses, falseAlarms, correctRejections))
        return sdtList


    # adding ROC plot method
    @staticmethod
    def plot_roc(sdtList):
        plt.plot([0,1], [0,1], 'k--', label= 'Chance')
        #variables to be used in plot
        for sdt in sdtList:
            hr = sdt.hit_rate()
            far = sdt.false_alarm_rate()
            plt.plot(far, hr, 'o', label=f'd prime={sdt.d_prime():.2f}', linewidth=2, markersize=8)
        plt.grid()
        plt.ylabel('Hit Rate')
        plt.xlabel('False Alarm Rate')
        plt.legend()
        plt.title('Receive Operating Characteristic')
        plt.show()


    def nLogLikelihood(self, hit_rate, false_alarm_rate):
        hit_rate = float(hit_rate)
        false_alarm_rate = float(false_alarm_rate)
        return - ((self.hits * np.log(hit_rate)) +
                  (self.misses * np.log(1 - hit_rate)) +
                  (self.falseAlarms * np.log(false_alarm_rate)) +
                  (self.correctRejections * np.log(1 - false_alarm_rate)))


    @staticmethod
    def rocCurve(false_alarm_rate, a):
        return stats.norm.cdf(a + stats.norm.ppf(false_alarm_rate))


    @staticmethod
    def rocLoss(a, sdt_list):
        """L = SignalDetection.rocLoss(a, sdtList)
        L(a) = sum_i[ell(Phi[a + Phi^{-1}(gamma_i)],gamma_i; h_i, f_i, m_i, r_i)]"""
        La = 0
        for sdt in sdt_list:
            La += SignalDetection.nLogLikelihood(sdt, SignalDetection.rocCurve
            (sdt.false_alarm_rate(), a), sdt.false_alarm_rate())
        return La


    @staticmethod
    def fit_roc(sdtList):
        a = 0
        aHat = 0
        for sdt in sdtList:
            SignalDetection.plot_roc(sdtList)
            # noinspection PyTypeChecker
            minimize = optimize.minimize(fun=sdt.rocLoss, args=sdtList, x0=a, method='BFGS')
            #fitting the function: minimizing a
            aHat = minimize.x
            x = np.linspace(0, 1, num=100)
            y = sdt.rocCurve(x, aHat)
            plt.plot(x, y, 'r-', linewidth=2, markersize=8)
        plt.ylabel('Hit Rate')
        plt.xlabel('False Alarm Rate')
        plt.title('Receive Operating Characteristic')
        plt.legend()
        return float(aHat)


    @staticmethod
    def plot_sdt(d_prime):
        c = d_prime / 2  # threshold value
        x = np.linspace(-4, 4, 1000)  # axes
        signal = stats.norm.pdf(x, loc=d_prime, scale=1) #signalcurve
        noise = stats.norm.pdf(x, loc=0, scale=1) #noisecurve

        # calculate max of signal and noise curves for d' line
        Nmax_y = np.max(noise)
        Nmax_x = x[np.argmax(noise)]
        Smax_y = np.max(signal)
        Smax_x = x[np.argmax(signal)]

        # plot curves

        plt.plot(x, signal, label='Noise')
        plt.plot(x, noise, label='Signal')
        plt.axvline(x=(d_prime / 2) + c, color='g', linestyle='--',
                    label='Threshold')  # vertical line over plot for d'/2+c
        plt.plot([Nmax_x, Smax_x], [Nmax_y, Smax_y], linestyle='--', lw=2, color='r', label = 'd prime')
        plt.legend()
        plt.xlabel('Stimulus intensity')
        plt.ylabel('Probability density')
        plt.title('Signal Detection Theory')
        plt.show()


