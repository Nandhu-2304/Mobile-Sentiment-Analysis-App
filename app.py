# app.py
import streamlit as st
import pandas as pd
from sentiment_model import analyze_sentiment_ml
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import os
import seaborn as sns

# --- App Configuration and Styling ---
st.set_page_config(
    page_title="Mobile Sentiment Analysis",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for a more attractive look
st.markdown(
    """
    <style>
    /* Main container and content styling */
    .stApp {
        background-color: #f0f2f6;
        color: #333333;
    }
    
    /* Title and Header */
    .st-emotion-cache-18ni295 { /* Targeting Streamlit's main title */
        font-family: 'Arial', sans-serif;
        font-weight: bold;
        color: #4a4a4a;
        text-align: center;
        padding-top: 20px;
        padding-bottom: 20px;
    }

    /* Sidebar Styling */
    .st-emotion-cache-12m1412 { /* Targeting Streamlit's sidebar */
        background-color: #ffffff;
        border-right: 1px solid #e0e0e0;
        padding-top: 50px;
    }
    
    /* Selectbox styling in the sidebar */
    .st-emotion-cache-13ln4jf {
        background-color: #ffffff;
        border: 1px solid #ccc;
        border-radius: 5px;
    }
    
    /* Button styling */
    .st-emotion-cache-1x4v529 { /* Targeting Streamlit buttons */
        background-color: #4a90e2;
        color: white;
        font-weight: bold;
        border: none;
        padding: 10px 20px;
        border-radius: 5px;
        cursor: pointer;
    }

    /* Subheader and markdown styling */
    h1, h2, h3 {
        color: #4a4a4a;
    }
    
    .st-markdown strong {
        color: #4a90e2;
    }
    
    /* Card-like containers for sections */
    .st-emotion-cache-f063m8 {
        background-color: #ffffff;
        border-radius: 10px;
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
        padding: 20px;
        margin-bottom: 20px;
    }

    </style>
    """,
    unsafe_allow_html=True
)

# --- Load dataset ---
if 'df' not in st.session_state:
    st.session_state.df = pd.read_csv("data/mobile_reviews_full.csv")

# --- Header with a sleek look ---
st.markdown(
    "<h1 style='text-align: center; color: #4a4a4a; font-size: 3em;'>📱 Mobile Reviews Sentiment Analysis</h1>", 
    unsafe_allow_html=True
)
st.markdown("---")

# Sidebar Menu
menu = ["View Ratings/Reviews", "Add Your Review", "Compare Two Mobiles", "Add New Model/Company"]
choice = st.sidebar.selectbox("Menu", menu)

# --- Common functions for the app ---
def generate_wordcloud(text):
    """Generates a word cloud from a given text string."""
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)
    fig, ax = plt.subplots()
    ax.imshow(wordcloud, interpolation='bilinear')
    ax.axis('off')
    st.pyplot(fig)

def plot_sentiment_distribution(data, title):
    """Plots a bar chart of sentiment distribution."""
    sentiment_counts = data['sentiment'].value_counts().reindex(['Positive', 'Neutral', 'Negative'], fill_value=0)
    fig, ax = plt.subplots()
    sns.barplot(x=sentiment_counts.index, y=sentiment_counts.values, ax=ax, palette=['#66c2a5', '#fc8d62', '#8da0cb'])
    ax.set_title(title, fontsize=15)
    ax.set_xlabel('Sentiment')
    ax.set_ylabel('Number of Reviews')
    st.pyplot(fig)

def display_mobile_image(mobile_name):
    """Displays a mobile image if it exists."""
    image_path = f"images/{mobile_name.replace(' ', '')}.jpg"
    if os.path.exists(image_path):
        st.image(image_path, width=250)
    else:
        st.info("Image not available for this model.")

# --- Sidebar logic ---
companies = st.session_state.df['company'].unique().tolist()
company_selected = st.sidebar.selectbox("Select Company", companies)
mobiles = st.session_state.df[st.session_state.df['company']==company_selected]['product_name'].unique().tolist()

# --- Main Page Logic with improved containers ---
if choice == "View Ratings/Reviews":
    with st.container():
        st.subheader("Explore Reviews and Insights")
        mobile_selected = st.selectbox("Select Mobile Model", mobiles)
        mobile_data = st.session_state.df[st.session_state.df['product_name']==mobile_selected]
        
        col_img, col_stats = st.columns([1, 2])
        
        with col_img:
            display_mobile_image(mobile_selected) # Call function here
        
        with col_stats:
            avg_rating = mobile_data['rating'].mean()
            st.metric(label="Average Rating", value=f"{avg_rating:.2f} ⭐")
            
            st.markdown("---")
            st.subheader("Sentiment Distribution")
            plot_sentiment_distribution(mobile_data, f"Sentiment for {mobile_selected}")

    with st.container():
        st.subheader("Top Reviews")
        col_pos, col_neg = st.columns(2)
        
        with col_pos:
            st.markdown("### ✅ Positive Reviews")
            for rev in mobile_data[mobile_data['sentiment']=="Positive"]['review_text'].head(5):
                st.success(f"• {rev}")
        
        with col_neg:
            st.markdown("### ❌ Negative Reviews")
            for rev in mobile_data[mobile_data['sentiment']=="Negative"]['review_text'].head(5):
                st.error(f"• {rev}")


elif choice == "Add Your Review":
    with st.container():
        st.subheader("Add Your Review")
        mobile_selected = st.selectbox("Select Mobile Model", mobiles)
        
        # Display image for the selected mobile
        display_mobile_image(mobile_selected)

        review_text = st.text_area("Enter your review (English + Telugu mix)")
        rating = st.slider("Rate the mobile", 1, 5)
        
        if st.button("Submit Review"):
            sentiment = analyze_sentiment_ml(review_text)
            st.info(f"Your review sentiment: **{sentiment}**")
            
            new_row = pd.DataFrame([{
                        "company": company_selected,
                        "product_name": mobile_selected,
                        "review_text": review_text,
                        "rating": rating,
                        "date": pd.to_datetime("today").date(),
                        "sentiment": sentiment
            }])
            
            # Using st.session_state to update the DataFrame
            st.session_state.df = pd.concat([st.session_state.df, new_row], ignore_index=True)
            st.success("Review added successfully! (Refresh to see changes)")


elif choice == "Compare Two Mobiles":
    with st.container():
        st.subheader("Compare Two Mobiles")
        all_mobiles = st.session_state.df['product_name'].unique().tolist()
        
        col1, col2 = st.columns(2)
        with col1:
            mobile1 = st.selectbox("Select First Mobile", all_mobiles, key='mobile1_comp')
            # Display image for the first mobile
            display_mobile_image(mobile1)
        with col2:
            mobile2 = st.selectbox("Select Second Mobile", all_mobiles, key='mobile2_comp')
            # Display image for the second mobile
            display_mobile_image(mobile2)
        
        if mobile1 and mobile2:
            data1 = st.session_state.df[st.session_state.df['product_name']==mobile1]
            data2 = st.session_state.df[st.session_state.df['product_name']==mobile2]

            st.markdown("---")
            st.subheader("Key Statistics")
            colA, colB = st.columns(2)
            with colA:
                st.metric(label=f"{mobile1} Avg Rating", value=f"{data1['rating'].mean():.2f} ⭐")
            with colB:
                st.metric(label=f"{mobile2} Avg Rating", value=f"{data2['rating'].mean():.2f} ⭐")

            st.markdown("---")
            st.subheader("Sentiment Comparison")
            colA, colB = st.columns(2)
            with colA:
                plot_sentiment_distribution(data1, f"Sentiment for {mobile1}")
            with colB:
                plot_sentiment_distribution(data2, f"Sentiment for {mobile2}")
            
            st.markdown("---")
            st.subheader("WordClouds")
            colA, colB = st.columns(2)
            with colA:
                st.write(f"**{mobile1} WordCloud**")
                generate_wordcloud(" ".join(data1['review_text']))
            with colB:
                st.write(f"**{mobile2} WordCloud**")
                generate_wordcloud(" ".join(data2['review_text']))
                
elif choice == "Add New Model/Company":
    with st.container():
        st.subheader("Add a New Mobile Model or Company")
        st.info("To add a new company, simply type the name below. If you select an existing company, a new model will be added to it.")
        
        # New company input
        new_company = st.text_input("Enter Company Name (e.g., 'Google')", key="new_company_input")
        if new_company:
            st.write(f"You will add a new model to: **{new_company}**")
            company_to_add = new_company
        else:
            # Or select an existing company
            company_to_add = st.selectbox("Select Existing Company", st.session_state.df['company'].unique().tolist())

        # New product name
        new_product_name = st.text_input("Enter New Mobile Model Name (e.g., 'Pixel 8')", key="new_product_name")
        first_review = st.text_area("Enter a sample review for this model", key="first_review")
        rating = st.slider("Rate the mobile", 1, 5, key="new_rating")
        
        # Image uploader for the new mobile
        uploaded_image = st.file_uploader("Upload an image for the new mobile (Optional)", type=["jpg", "jpeg", "png"])
        
        if st.button("Submit"):
            if new_product_name and first_review:
                sentiment = analyze_sentiment_ml(first_review)
                
                # Create the new row
                new_row = pd.DataFrame([{
                            "company": company_to_add,
                            "product_name": new_product_name,
                            "review_text": first_review,
                            "rating": rating,
                            "date": pd.to_datetime("today").date(),
                            "sentiment": sentiment
                }])
                
                # Update the session state DataFrame
                st.session_state.df = pd.concat([st.session_state.df, new_row], ignore_index=True)
                
                # Permanently save the updated DataFrame to the CSV file
                st.session_state.df.to_csv("data/mobile_reviews_full.csv", index=False)

                # Save the uploaded image permanently
                if uploaded_image:
                    image_dir = "images"
                    if not os.path.exists(image_dir):
                        os.makedirs(image_dir)
                    
                    # Construct image file path
                    image_filename = f"{new_product_name.replace(' ', '')}.jpg"
                    image_path = os.path.join(image_dir, image_filename)
                    
                    with open(image_path, "wb") as f:
                        f.write(uploaded_image.getbuffer())
                    
                    st.success(f"New mobile '{new_product_name}' and its image added successfully!")
                else:
                    st.success(f"New mobile '{new_product_name}' added successfully!")
            else:
                st.warning("Please enter a mobile name and a sample review.")