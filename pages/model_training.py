import streamlit as st
from data_utils import prepare_features
from visualizations import plot_actual_vs_predicted
from model_utils import split_data, load_model, evaluate_model, save_model, train_model



def show(df):
    """Display the model training page"""
    st.header("Model Training")

    # Prepare feature  and target
    X, y = prepare_features(df)

    # Split data into training and test sets
    test_size = st.slider("Test data size (%)", 10, 40, 20) / 100
    X_train, X_test, y_train, y_test = split_data(X, y, test_size)

    st.write(f"Training Data: {len(X_train)} samples")
    st.write(f"Testing Data: {len(X_test)} samples")

    model_type = st.selectbox("Select the model Types", ['"Linear Regression', 'Random Forest'])

    # Train the model - button to train the model
    if st.button("Train Model"):
        with st.spinner("Training in progress..."):
            # Train the model
            model = train_model(X_train, y_train, model_type)

            # Evaluate teh model
            metrics = evaluate_model(model, X_train, y_train, X_test, y_test)

            # Display  the metrics
            col1, col2 = st.columns(2)
            with col1:
                st.subheader("Training Metrics")
                st.write(f"RMSE : {metrics['train_rmse']:.2f} C")
                st.write(f"R : {metrics['train_r2']:.4f}")

            with col2:
                st.subheader("Testing Metrics")
                st.write(f"RMSE : {metrics['test_rmse']:.2f} C")
                st.write(f"R : {metrics['test_r2']:.4f}")
            
            # Plot the actual vs the predicted 
            st.subheader("Actual vs Predicted (test data)")
            fig = plot_actual_vs_predicted(metrics['y_test'], metrics['y_pred_test'])
            st.pyplot(fig)

            # Save the model
            save_model(model)

            st.success("Model trained and saved successfully!")
            st.session_state['model'] = model
            st.session_state['model_type'] = model_type
