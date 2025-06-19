
import streamlit as st
import pickle               #loads the trained model
import numpy as np
from scipy.spatial import distance        #distance (from scipy.spatial):Computes distances between points for cluster assignment.

with open("segmentation_model", "rb") as f:
    model = pickle.load(f)         #Uses pickle.load(f) to deserialize the saved DBSCAN model.

#Defining a Function to Predict Cluster Assignments
def dbscan_predict(dbscan_model, X_new):  
    #Creates an array y_new initialized with -1 (noise points) for all incoming customer data.
    y_new = np.ones(shape=len(X_new), dtype=int) * -1  
    for j, x_new in enumerate(X_new):              #X_new: new data points, Compares each input (x_new) against core samples
        for i, x_core in enumerate(dbscan_model.components_):
            if distance.euclidean(x_new, x_core) < dbscan_model.eps:
                y_new[j] = dbscan_model.labels_[dbscan_model.core_sample_indices_[i]]
                break
    return y_new                 #Returns the predicted cluster labels.


# Creating the Streamlit Web App Interface
st.title("🧩 Customer Segmentation with DBSCAN")
st.markdown("Enter customer features below to predict their **segment** based on spending behavior.")

# Input sliders: 
income = st.slider("Annual Income (normalized)", 0.0, 25.0, 10.0)
mnt_wines = st.slider("Spending on Wine Products", -1.0, 0.0, -0.88)
mnt_meat = st.slider("Spending on Meat Products", -1.0, 6.0, -0.7)
mnt_fruits = st.slider("Spending on Fruits", -1.0, 0.0, -0.5)
mnt_fish = st.slider("Spending on Fish Products", -1.0, 0.0, -0.6)
mnt_sweets = st.slider("Spending on Sweet Products", -1.0, 6.0, 2.0)
mnt_gold = st.slider("Spending on Gold Products", -1.0, 6.0, 1.5)

# Converts the user-selected values into a NumPy array, suitable for clustering.
X_input = np.array([[income, mnt_wines, mnt_meat, mnt_fruits, mnt_fish, mnt_sweets, mnt_gold]])


# Running Predictions & Displaying Results
if st.button("Predict Segment"):
    cluster = dbscan_predict(model, X_input)[0]      #Calls dbscan_predict(model, X_input) to assign a customer segment
    st.success(f"Predicted Cluster: {cluster}")      #Displays the result

    
    if cluster == -1:
        st.markdown("💎 **Luxury Sweet Spot Customers**: High spenders on sweets & gold. Likely indulgent and wealthy.")
    elif cluster == 0:
        st.markdown("📦 **Balanced Shoppers**: Average spenders across categories—steady and practical.")
    elif cluster == 1:
        st.markdown("🥩 **Meat-Lovers Segment**: High spend on meat and proteins—likely bulk buyers.")
    else:
        st.warning("Cluster not recognized. Try adjusting input or recheck the model.")
