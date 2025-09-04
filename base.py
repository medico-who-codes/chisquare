import streamlit as st
import numpy as np
from scipy.stats import chi2_contingency
import pandas as pd

# Streamlit app title
st.title("Chi-Square Test")

# Initialize original data
original_data = {
    "A": {"Malignant": 41, "Benign": 7, "Total": 48},
    "B": {"Malignant": 108, "Benign": 48, "Total": 156},
    "C": {"Malignant": 132, "Benign": 51, "Total": 183},
    "D": {"Malignant": 33, "Benign": 12, "Total": 45}
}
original_proportions = {
    "A": 0.1144,
    "B": 0.3611,
    "C": 0.4236,
    "D": 0.1042
}
original_total = 432

# Slider for total sample size
st.subheader("Set Total Sample Size (400–464)")
total_sample_size = st.slider(
    "Total Sample Size",
    min_value=400,
    max_value=464,
    value=432,
    step=1,
    key="total_sample_size"
)

# Calculate target row totals based on proportions
row_totals = {
    pheno: round(total_sample_size * prop)
    for pheno, prop in original_proportions.items()
}
# Adjust to ensure sum matches total_sample_size
total_calculated = sum(row_totals.values())
if total_calculated != total_sample_size:
    diff = total_sample_size - total_calculated
    row_totals["C"] += diff

# Initialize session state for counts
if "counts" not in st.session_state:
    st.session_state.counts = {
        pheno: {
            "Malignant": original_data[pheno]["Malignant"],
            "Benign": original_data[pheno]["Benign"]
        }
        for pheno in original_data
    }

# Function to update counts and ensure row totals
def update_counts(pheno, col):
    # Get the slider value from session_state using the slider's key
    value = st.session_state[f"slider_{pheno}_{col}"]
    st.session_state.counts[pheno][col] = value
    # Adjust the other column to maintain row total
    other_col = "Benign" if col == "Malignant" else "Malignant"
    st.session_state.counts[pheno][other_col] = (
        row_totals[pheno] - st.session_state.counts[pheno][col]
    )

# Create sliders for each cell
st.subheader("Adjust Contingency Table")
cols = st.columns([1, 2, 2, 1])  # Columns for layout: Phenotype, Malignant, Benign, Total
with cols[0]:
    st.write("**Phenotype**")
    for pheno in original_data:
        st.write(pheno)
with cols[1]:
    st.write("**Malignant**")
    for pheno in original_data:
        value = st.slider(
            f"{pheno}_Malignant",
            min_value=0,
            max_value=row_totals[pheno],
            value=st.session_state.counts[pheno]["Malignant"],
            step=1,
            key=f"slider_{pheno}_Malignant",
            on_change=update_counts,
            args=(pheno, "Malignant")
        )
with cols[2]:
    st.write("**Benign**")
    for pheno in original_data:
        value = st.slider(
            f"{pheno}_Benign",
            min_value=0,
            max_value=row_totals[pheno],
            value=st.session_state.counts[pheno]["Benign"],
            step=1,
            key=f"slider_{pheno}_Benign",
            on_change=update_counts,
            args=(pheno, "Benign")
        )
with cols[3]:
    st.write("**Total**")
    for pheno in original_data:
        st.write(row_totals[pheno])

# Create contingency table for chi-square test
contingency_table = np.array([
    [st.session_state.counts[pheno]["Malignant"], st.session_state.counts[pheno]["Benign"]]
    for pheno in original_data
])

# Perform chi-square test
try:
    chi2, p, dof, expected = chi2_contingency(contingency_table, correction=False)
except ValueError as e:
    chi2, p, dof = np.nan, np.nan, 3
    expected = np.zeros_like(contingency_table)
    st.error(f"Chi-square test failed: {e}. Ensure all expected frequencies are positive.")

# Display results
st.subheader("Contingency Table")
df = pd.DataFrame(
    contingency_table,
    index=["A", "B", "C", "D"],
    columns=["Malignant", "Benign"]
)
df["Total"] = df.sum(axis=1)
df.loc["Total"] = df.sum(axis=0)
st.dataframe(df.style.format("{:.0f}"))

st.subheader("Chi-Square Test Results")
st.write(f"Chi-Square Statistic: {chi2:.3f}")
st.write(f"P-Value: {p:.4f}")
st.write(f"Degrees of Freedom: {dof}")
st.write(f"Significance (α = 0.05): {'Significant' if p < 0.05 else 'Not Significant'}")

st.subheader("Expected Frequencies")
df_expected = pd.DataFrame(
    expected,
    index=["A", "B", "C", "D"],
    columns=["Malignant", "Benign"]
)
st.dataframe(df_expected.style.format("{:.2f}"))

# Display malignancy rates
st.subheader("Malignancy Rates (%)")
malignancy_rates = {
    pheno: (st.session_state.counts[pheno]["Malignant"] / row_totals[pheno] * 100)
    for pheno in original_data
}
for pheno, rate in malignancy_rates.items():
    st.write(f"{pheno}: {rate:.2f}%")
st.write(f"Overall: {contingency_table[:, 0].sum() / total_sample_size * 100:.2f}%")