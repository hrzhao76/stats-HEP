import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
from tqdm import tqdm
import hist
import mplhep
from typing import Union

mplhep.style.use(mplhep.style.ATLAS)

np.random.seed(42)


def Likelihood(
    n: Union[int, np.array],
    m: Union[int, np.array],
    b: int,
    mu: float,
    s: int = 10,
    tau: float = 1,
) -> Union[float, np.array]:
    """The Likelihood function for the Counting experiment, eqn 90.

    Args:
        n (Union[int, np.array]): Observed number of events.
        m (Union[int, np.array]): Control sample help to constrain the nuisance parameter b.
        b (int): Expected number from background processes.
        mu (float): Signal strength.
        s (int, optional): Expected number of signal events. Defaults to 10.
        tau (float, optional): Scale factor. Defaults to 1.

    Returns:
        Union[float, np.array]: The likelihood function.
    """
    poisson_sig = stats.poisson.pmf(mu=mu * s + b, k=n)
    poisson_bkg = stats.poisson.pmf(mu=tau * b, k=m)
    return poisson_sig * poisson_bkg


def MLE_mu_hat(
    n: Union[int, np.array],
    m: Union[int, np.array],
    s: int = 10,
    tau: float = 1,
) -> Union[float, np.array]:
    """Maximum likelihood estimator of mu, eqn 91.

    Args:
        n (Union[int, np.array]): Observed number of events.
        m (Union[int, np.array]): Control sample help to constrain the nuisance parameter b.
        s (int, optional): Expected number of signal events. Defaults to 10.
        tau (float, optional): Scale factor. Defaults to 1.

    Returns:
        Union[float, np.array]: The MLE of mu.
    """
    return (n - m / tau) / (s)


def MLE_b_hat(
    m: Union[int, np.array],
    tau: float = 1,
) -> Union[float, np.array]:
    """Maximum likelihood estimator of b, eqn 92.

    Args:
        m (Union[int, np.array]): Control sample help to constrain the nuisance parameter b.
        tau (float, optional): Scale factor. Defaults to 1.

    Returns:
        Union[float, np.array]: The MLE of b.
    """
    return m / tau


def MLE_b_hat_hat(
    n: Union[int, np.array],
    m: Union[int, np.array],
    mu: float,
    s: int,
    tau: float = 1,
) -> Union[float, np.array]:
    """Conditional maximum likelihood estimator of b, eqn 93.

    Args:
        n (Union[int, np.array]): Observed number of events.
        m (Union[int, np.array]): Control sample help to constrain the nuisance parameter b.
        mu (float): A specified μ in conditional likelihood.
        s (int): Expected number of signal events.
        tau (float, optional): Scale factor. Defaults to 1.

    Returns:
        Union[float, np.array]: The conditional MLE of b.
    """
    a = tau + 1
    b = mu * s + tau * mu * s - m - n
    c = -m * mu * s

    return (-b + np.sqrt(np.power(b, 2) - 4 * a * c)) / (2 * a)


def Lambda_mu(
    n: Union[int, np.array], m: Union[int, np.array], mu: float, s: int, tau: float = 1
) -> Union[float, np.array]:
    """The Likelihood ratio test statistic, eqn 7.

    Args:
        n (Union[int, np.array]): Observed number of events.
        m (Union[int, np.array]): Control sample help to constrain the nuisance parameter b.
        mu (float): A specified μ in conditional likelihood.
        s (int): Expected number of signal events.
        tau (float, optional): Scale factor. Defaults to 1.

    Returns:
        Union[float, np.array]: The Likelihood ratio test statistic.
    """
    profiled_likelihood = Likelihood(n=n, m=m, mu=mu, b=MLE_b_hat_hat(n, m, mu, s))
    unconditional_likelihood = Likelihood(n=n, m=m, mu=MLE_mu_hat(n, m), b=MLE_b_hat(m))
    return profiled_likelihood / unconditional_likelihood


def Get_q0(
    n: Union[int, np.array], m: Union[int, np.array], mu: float, s: int, tau: float = 1
) -> Union[float, np.array]:
    """Calculate the test statistic q0 for discovery of a positive signal, eqn 12.

    Args:
        n (Union[int, np.array]): Observed number of events.
        m (Union[int, np.array]): Control sample help to constrain the nuisance parameter b.
        mu (float): A specified μ in conditional likelihood.
        s (int): Expected number of signal events.
        tau (float, optional): Scale factor. Defaults to 1.

    Returns:
        Union[float, np.array]: The test statistic q0 for the sample.
    """

    mu_hat = MLE_mu_hat(n=n, m=m, tau=tau)
    neg_index = mu_hat < 0

    lambda_mu = Lambda_mu(n=n, m=m, mu=0, s=s)
    lambda_mu[neg_index] = 1
    q0 = -2 * np.log(lambda_mu)

    return q0


def Get_qmu(
    n: Union[int, np.array], m: Union[int, np.array], mu: float, s: int, tau: float = 1
) -> Union[float, np.array]:
    """Calculate the test statistic qmu for upper limits, eqn 14.

    Args:
        n (Union[int, np.array]): Observed number of events.
        m (Union[int, np.array]): Control sample help to constrain the nuisance parameter b.
        mu (float): A specified μ in conditional likelihood.
        s (int): Expected number of signal events.
        tau (float, optional): Scale factor. Defaults to 1.

    Returns:
        Union[float, np.array]: The test statistic qmu.
    """
    mu_hat = MLE_mu_hat(n=n, m=m, tau=tau)
    greater_index = mu_hat > mu

    lambda_mu = Lambda_mu(n=n, m=m, mu=mu, s=s)
    lambda_mu[greater_index] = 1
    qmu = -2 * np.log(lambda_mu)

    return qmu


def f_q0(q0: Union[float, np.array]) -> Union[float, np.array]:
    """pdf of q0 given mu_prime=0, eqn 49.

    Args:
        q0 (Union[float, np.array]): Test statistic q0 for discovery.

    Returns:
        Union[float, np.array]: pdf of q0.
    """
    return 1 / 2 * 1 / np.sqrt(2 * np.pi) * 1 / np.sqrt(q0) * np.exp(-q0 / 2)


def f_q0_muprime(
    q0: Union[float, np.array], mu_prime: float, sigma: float
) -> Union[float, np.array]:
    """pdf of q0 given mu_prime, eqn 48.

    Args:
        q0 (Union[float, np.array]): Test statistic q0 for discovery.
        mu_prime (float): The signal strength.

    Returns:
        Union[float, np.array]: pdf of q0.
    """
    return (
        0.5
        / np.sqrt(2 * np.pi)
        / np.sqrt(q0)
        * np.exp(-0.5 * np.power(np.sqrt(q0) - mu_prime / sigma, 2))
    )


def calSigma_Asimov(mu_prime: float, s: int, b: int, mu: float) -> float:
    """Calculate the standard deviation of mu_hat assuming a strength parameter mu_prime using
    Asimov data, Eqn 31, 32.

    Args:
        mu_prime (float): The signal strength.
        s (int): Expected number of signal events.
        b (int): Expected number of background events.
        mu (float): A specified μ in conditional likelihood.

    Returns:
        float: The standard deviation of mu_hat.
    """
    Likelihood_A = Lambda_mu(n=mu_prime * s + b, m=b, mu=mu, s=s)
    q_mu_A = -2 * np.log(Likelihood_A)
    sigma = 1.0 / np.sqrt(q_mu_A)

    return sigma


def secondDerivativeMu(n, s, b, mu):
    denominator = n * np.power(s, 2)
    numerator = np.power((mu * s + b), 2)
    return -1 * denominator / numerator


def secondDerivativeB(n, m, s, b, mu):
    first_term = m / np.power(b, 2)
    second_term = n / np.power((mu * s + b), 2)
    return -1 * (first_term + second_term)


def mixedSecondDerivativeMuB(n, s, b, mu):
    denominator = n * s
    numerator = np.power((mu * s + b), 2)
    return -1 * denominator / numerator


def calSigma_Cov(
    s: int, b: int, tau: float, mu_prime: float, N_trails=10000000
) -> float:
    """Calculate the standard deviation of mu_hat assuming a strength parameter mu_prime using
    covariance matrix, Eqn 18.

    Args:
        s (int): Expected number of signal events.
        b (int): Expected number of background events.
        tau (float): Scale factor.
        mu_prime (float): The signal strength.
        N_trails (int, optional): Number of data generated to estimtate. Defaults to 10000000.

    Returns:
        float: The standard deviation of mu_hat.
    """
    n_s = stats.poisson.rvs(mu=mu_prime * s + b, size=N_trails)
    m_s = stats.poisson.rvs(mu=tau * b, size=N_trails)

    V_inv_00 = -np.mean(secondDerivativeMu(n=n_s, s=s, b=b, mu=mu_prime))
    V_inv_11 = -np.mean(secondDerivativeB(n=n_s, m=m_s, s=s, b=b, mu=mu_prime))
    V_inv_01 = -np.mean(mixedSecondDerivativeMuB(n=n_s, s=s, b=b, mu=mu_prime))
    V_inv_10 = V_inv_01

    V_inv = np.array([[V_inv_00, V_inv_01], [V_inv_10, V_inv_11]])
    V = np.linalg.inv(V_inv)
    sigma_mu = np.sqrt(V[0, 0])
    return sigma_mu


def calZ0(p):
    return stats.norm.isf(p)


def calPvalue(Z0):
    return stats.norm.sf(Z0)


def drawnm(n_s: np.array, m_s: np.array) -> (plt.Figure, plt.Axes):
    n_bins = 50
    hist_2d = hist.Hist(
        hist.axis.Regular(n_bins, 0, n_bins, name="n_s"),
        hist.axis.Regular(n_bins, 0, n_bins, name="m_s"),
    )

    hist_2d.fill(n_s, m_s)

    fig, axs = plt.subplots(1, 2, figsize=(9, 4))
    mplhep.histplot(hist_2d.project("n_s"), label="n_s", ax=axs[0])
    mplhep.histplot(hist_2d.project("m_s"), label="m_s", ax=axs[1])
    return fig, axs
