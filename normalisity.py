import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import skew, kurtosis, shapiro, probplot
import seaborn as sns

add_selectbox = st.sidebar.header("Normalisity Analysing Web APP")
st.sidebar.write("Created by Milad.")
st.sidebar.write("Source at [My GitHub](https://github.com/miladrayka/normalisity_ed/)")

sns.set_style("darkgrid")

st.title("Investigating Normalisity of an Error Distribution")

st.write(
    """If the distribution of residual errors for a benchmark dataset is zero-centered normal, statistics such as
the mean absolute error (*MAE*), the root mean squared deviation (*RMSD*) or the $95^{th}$ quantile of the
absolute errors distribution ($Q_{95}$) are redundant and can be used to infer *u(M)*:
*u(M)* ≃ *RMSD* = √π /2*MAE* ≃ 0 .5$Q_{95}$. If it is not normal, more information is necessary to provide the
user with probabilistic diagnostics."""
)

st.header("Workflow")

st.write(
    """Statistical benchmarking of a method *M* is based on the estimation of errors 
(*EM* = {$e_{M, i}$| *i*=1,...,*N*}) for a set of *N* calculated (*CM* = {$C_{M, i}$| *i*=1,...,*N*}) and reference data 
(*R* = {$r_{i}$| *i*=1,...,*N*}), where:\n
$e_{M, i} = r_{i} - C_{M,i}$"""
)

option = st.selectbox("Choose one:", ("Example", "My Data"))

if option == "Example":

    number = st.number_input(
        "Choose number of data points", min_value=100, max_value=None
    )
    x_true = np.random.randn(int(number), 1)

    st.write(f"Generate {int(number)} data points randomly from a normal distribution.")

    y_true = x_true * 2 + 5
    st.write("Target values are generated using:     $y = x*2 + 5$.")

    y_predict = x_true * 1.8 + 4
    st.write("But, predicted values have derived from:   $y = x*1.8 + 4$.")

    df = pd.DataFrame(
        {"Target": y_true.ravel(), "Predicted": y_predict.ravel()},
        index=range(len(y_true)),
    )

if option == "My Data":

    st.write(
        """Caution: Your csv file should has two columns with the following names: 
             **Target** and **Predicted**."""
    )

    uploaded_file = st.file_uploader("Load your data", type="csv")
    df = pd.read_csv(uploaded_file)


st.dataframe(df)

residual_error = df["Target"] - df["Predicted"]

mse = np.mean(residual_error)
rmsd = np.sqrt(np.mean((residual_error) ** 2))
mae = np.mean(np.abs(residual_error))

st.write(f"Mean square error (MSE): {mse:.3f}")
st.write(f"Root mean square deviation (RMSD): {rmsd:.3}")
st.write(f"Mean absolute error (MAE): {mae:.3f}")

st.subheader("Shapiro-Wilk (W) Test")

st.write(
    """For a given sample size, the Shapiro-Wilk *W* statistics has been shown to 
have good properties. The values of *W* range between 0 and 1, and values of *W* ≃ 1 are in 
favor of the normality of the sample. If *W* lies below a critical value $W_{c}$
depending on the sample size and the chosen level of type I errors *α* (typically
0.05), the normality hypothesis cannot be assumed to hold."""
)

shapiro_wilk_w, p_value = shapiro(residual_error)

col1, col2 = st.columns(2)
col1.metric("W", np.round(shapiro_wilk_w, 4))
col2.metric("Pr", np.round(p_value, 3))

if 0.05 < p_value:
    st.write("So, error distribution is normal (*Pr* > 0.05).")

st.subheader("Skewness and Kurtosis Tests")

st.write(
    """Two other statistics are helpful in characterizing the departure from normality. The skewness (*Skew*), or
third standardized moment of the distribution, quantifies its asymmetry (*Skew* = 0 for a symmetric
distribution). The kurtosis (*Kurt*), or fourth standardized moment, quantifies the concentration of data in
the tails of the distribution. Kurtosis of a normal distribution is equal to 3; distributions with excess kurtosis
(*Kurt > 3*) are called *leptokurtic*; those with *Kurt < 3* are named *platykurtic*."""
)

skewness_value = skew(residual_error)
kurtosis_value = kurtosis(residual_error, fisher=False)

col1, col2 = st.columns(2)
col1.metric("Skew", np.round(skewness_value, 4))
col2.metric("Kurtosis", np.round(kurtosis_value, 4))

if skewness_value != 0 and kurtosis_value > 3:
    st.write("Error distribution is asymmetric and leptokurtic.")

if skewness_value != 0 and kurtosis_value < 3:
    st.write("Error distribution is asymmetric and platykurtic.")

st.subheader("$95^{th}$ quantile of the absolute errors distribution")

st.write(
    """For error distributions which are non symmetric (*Skew* not 0), quantifying the accuracy by a single
dispersion-related statistic is not reliable, and one should provide probability intervals or accept to lose
information on the sign and use a statistic based on absolute errors, such as $Q_{95}$
(the 95th percentile of the absolute error distribution, gives the amplitude of errors that there is a 5%
probability to exceed)."""
)

q95_value = np.percentile(np.abs(residual_error), 95)

st.metric("Q95", np.round(q95_value, 4))

st.subheader("Normal Quantile-Quantile Plot")

st.write(
    """It might also be useful to assess normality by visual tools: normal quantile-quantile plots (*QQ-plots*),
where the quantiles of the scaled and centered errors sample is plotted against the theoretical quantiles of a
standard normal distribution (in the normal case, all points should lie over the unit line);"""
)


def plot_qqplot(residual, color="g", save=False):

    fig = plt.figure(figsize=(10, 8))
    x, y = range(-4, 5), range(-4, 5)
    quantile = probplot(residual, plot=None, fit=False)
    plt.plot(x, y, "--", label="Standard Normal Distribution")
    plt.plot(quantile[0], quantile[1], color, label="Sample")
    plt.xlim(-4, 4)
    plt.ylim(-4, 4)
    plt.xlabel("Theoretical Quantile", fontsize=14)
    plt.ylabel("Sample Quantile", fontsize=14)
    plt.title("Normal Quantile-Quantile Plot", fontsize=16)
    if save:
        plt.savefig("qqplot.png", dpi=300)
    plt.legend()
    plt.show()

    return fig


save = st.checkbox("Save QQ-plot in .png.")

color = st.color_picker("Choose color for QQ-plot:", value="#66CDAA")

fig = plot_qqplot(residual_error, color, save)

st.pyplot(fig)

st.subheader("Histogram of Error Distribution")

st.write(
    """comparison of the histogram of errors with a gaussian curve having the same mean, 
estimated by the mean signed error(*MSE*), and same standard deviation, estimated by the *RMSD*."""
)


def histogram_and_normal_plot(
    mse, rmsd, residual, color1="darkcyan", color2="orangered", save=False
):

    mu, sigma = mse, rmsd
    fig = plt.figure(figsize=(10, 8))
    _, bins, _ = plt.hist(residual, bins=20, density=True, color=color1)
    plt.plot(
        bins,
        1
        / (sigma * np.sqrt(2 * np.pi))
        * np.exp(-((bins - mu) ** 2) / (2 * sigma ** 2)),
        linewidth=2,
        color=color2,
    )
    plt.xlabel("Residual Error", fontsize=14)
    plt.ylabel("Density", fontsize=14)
    plt.title("Histogram of Residual Errors", fontsize=16)
    if save:
        plt.savefig("residual_histogram.png", dpi=300)
    plt.show()

    return fig


save = st.checkbox("Save histogram plot in .png.")

color1 = st.color_picker("Choose color for histogram plot:", value="#5F9EA0")
color2 = st.color_picker("Choose color for guassian plot:", value="#DC143C")


fig_hist = histogram_and_normal_plot(mse, rmsd, residual_error, color1, color2, save)

st.pyplot(fig_hist)

st.subheader("Outliers Plot")

st.write(
    """There is no unique method to identify outliers for a non-normal distribution. One might, for instance, use
visual tools, such as *QQ-plots*, or automatic selection tools, such as selecting points for which the
absolute error is larger than the 95th percentile ($Q_{95}$), or another percentile corresponding to a predefined
error threshold."""
)


def outliers_plot(
    nonoutliers_df, outliers_df, color1="deepskyblue", color2="firebrick", save=False
):

    fig = plt.figure(figsize=(10, 8))
    plt.scatter(nonoutliers_df["Target"], nonoutliers_df["Residual"], s=50, c=color1)
    plt.scatter(
        outliers_df["Target"], outliers_df["Residual"], s=50, c=color2, marker="^"
    )
    plt.xlabel("Experimental", fontsize=14)
    plt.ylabel("Residual Error", fontsize=14)
    plt.title("Error Distribution", fontsize=16)
    if save:
        plt.savefig("error_distribution.png.", dpi=300)
    plt.show()

    return fig


percentile = st.slider("Choose a specific percentile", step=5)

df["ABS"] = (df["Predicted"] - df["Target"]).abs()
df["Residual"] = residual_error
q_value = np.percentile(np.abs(residual_error), int(percentile))
outliers_df = df[df["ABS"] > q_value]
nonoutliers_df = df.drop(outliers_df.index, axis=0)

save = st.checkbox("Save outliers plot in .png")

color1 = st.color_picker("Choose color for non-outliers:", value="#00CDCD")
color2 = st.color_picker("Choose color for outliers:", value="#FF4040")

fig_outliers = outliers_plot(nonoutliers_df, outliers_df, color1, color2, save)


st.pyplot(fig_outliers)

st.subheader("References")
st.write(
    """All texts are extracted from *Mach. Learn.: Sci. Technol., 2020, 1, 035011* paper."""
)
