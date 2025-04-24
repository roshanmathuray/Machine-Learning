import os
import pandas as pd
import streamlit as st

# Handle missing google.colab import gracefully
try:
    import google.colab
except ImportError:
    pass  # Continue if Google Colab is not available

# File uploader to upload the CSV
uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])

if uploaded_file is not None:
    try:
        # Read the uploaded file
        df = pd.read_csv(uploaded_file)

        # Check for missing columns and handle them gracefully
        expected_columns = [
            "Income", "Recency", "MntWines", "MntFruits", "MntMeatProducts", "MntFishProducts", 
            "MntSweetProducts", "MntGoldProds", "NumWebPurchases", "NumStorePurchases", 
            "NumCatalogPurchases", "NumWebVisitsMonth", "NumDealsPurchases"
        ]

        # Check and display missing columns
        missing_columns = [col for col in expected_columns if col not in df.columns]
        if missing_columns:
            st.warning(f"Warning: Missing columns in the uploaded file: {', '.join(missing_columns)}")

        # Optionally, fill missing columns with default values or skip processing
        for col in missing_columns:
            df[col] = 0  # Fill missing columns with zero or use other appropriate handling
        
        # Display the uploaded DataFrame
        st.write(df)

        # Continue with your processing logic here...
        # For example, GMM or BIRCH algorithm could go here (just as a placeholder)
        # model = SomeModel().fit(df)

    except Exception as e:
        st.error(f"An error occurred: {e}")
else:
    st.info("Please upload a CSV file.")

# Check for the script file existence in the correct directory
script_path = "/mount/src/machine-learning/gmm_and_birch_for_customer_identity_and_brand_loyalty.py"
if os.path.exists(script_path):
    st.success("Main script found and ready to run.")
else:
    st.error(f"Main script not found at: {script_path}")
