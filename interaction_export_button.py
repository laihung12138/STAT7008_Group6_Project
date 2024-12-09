#!/usr/bin/env python
# coding: utf-8

# In[1]:


import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from io import BytesIO
import base64


# In[2]:


class DataExporter:
    def __init__(self, obj):
        self.obj = obj

    def convert_df_to_csv(self, df):
        return df.to_csv(index=False).encode('utf-8')

    def convert_df_to_excel(self, df):
        output = BytesIO()
        with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
            df.to_excel(writer, index=False, sheet_name='Sheet1')
        return output.getvalue()

    def display_options(self):
        if "export_option" not in st.session_state:
            st.session_state.export_option = None

        option = st.radio(
            "Choose export option",
            ["Copy Data", "Export as CSV", "Export as XLSX"],
            key="export_radio"
        )
        st.session_state.export_option = option

        if option == "Copy Data":
            if st.button("Copy to Clipboard"):
                self.obj.to_clipboard(index=False)
                st.success("Data copied to clipboard!")

        elif option == "Export as CSV":
            csv_data = self.convert_df_to_csv(self.obj)
            st.download_button(
                label="Download CSV",
                data=csv_data,
                file_name='data.csv',
                mime='text/csv'
            )

        elif option == "Export as XLSX":
            excel_data = self.convert_df_to_excel(self.obj)
            st.download_button(
                label="Download Excel",
                data=excel_data,
                file_name='data.xlsx',
                mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
            )


# In[ ]:


if __name__ == "__main__":
    # Sample Data
    df = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]})

    # Sample Plot
    fig, ax = plt.subplots()
    df.plot(kind='bar', ax=ax)
    plt.title("Sample Plot")

    # Test with DataFrame
    st.write("Export DataFrame:")
    exporter = DataExporter(df)
    exporter.display_options()
