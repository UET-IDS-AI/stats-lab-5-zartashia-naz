import numpy as np


# -------------------------------------------------
# Question 1 – Exponential Distribution
# -------------------------------------------------

def exponential_pdf(x, lam=1):
    """
    Return PDF of exponential distribution.

    f(x) = lam * exp(-lam*x) for x >= 0
    """
    if x < 0:
        return 0
    return lam * np.exp(-lam * x)


def exponential_interval_probability(a=2, b=5, lam=1):
    """
    Compute P(a < X < b) using analytical formula.

    P(a < X < b) = e^(-lam*a) - e^(-lam*b)
    """
    return np.exp(-lam * a) - np.exp(-lam * b)


def simulate_exponential_probability(a=2, b=5, n=100000):
    """
    Simulate exponential samples and estimate
    P(a < X < b).
    """
    samples = np.random.exponential(scale=1, size=n)
    return np.mean((samples > a) & (samples < b))


# -------------------------------------------------
# Question 2 – Bayesian Classification
# -------------------------------------------------

def gaussian_pdf(x, mu, sigma):
    """
    Return Gaussian PDF.
    """
    return 1 / (np.sqrt(2 * np.pi) * sigma) * np.exp(-(x - mu) ** 2 / (2 * sigma ** 2))


def posterior_probability(time=42):
    """
    Compute P(B | X = time)
    using Bayes rule.

    Priors:
    P(A)=0.3
    P(B)=0.7

    Distributions:
    A ~ N(40,4)
    B ~ N(45,4)
    """
    pA = 0.3
    pB = 0.7
    # sigma is same for both groups , so normalizing constant is cancel 
    fA = np.exp(-(time - 40) ** 2 / 4)
    fB = np.exp(-(time - 45) ** 2 / 4)
    return (pB * fB) / (pA * fA + pB * fB)


def simulate_posterior_probability(time=42, n=100000):
    """
    Estimate P(B | X=time) using simulation.
    """
    sigma = 2
    groups = np.random.choice(['A', 'B'], size=n, p=[0.3, 0.7])
    times = np.where(
        groups == 'A',
        np.random.normal(40, sigma, n),
        np.random.normal(45, sigma, n)
    )
    tolerance = 0.5
    mask = np.abs(times - time) < tolerance
    if mask.sum() == 0:
        return None
    return np.sum((groups == 'B') & mask) / mask.sum()
