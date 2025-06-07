import streamlit as st

#main_page = st.Page("chat.py", title="Main App", icon=":material/add_circle:")
#st.logo("logo.png", size="large")

main_page = st.Page("app.py", title="🏠 Main App")        
#predict = st.Page("predict.py", title="🧠 Prediction")    
#grafik = st.Page("grafik.py", title="📊 Grafik")          
#debug = st.Page("base.py", title="🛠️ Debug")
#DigTwin = st.Page("masking.py", title="🛠️ DigTwin") 

pg = st.navigation([main_page 
                    #predict,
                    #grafik,
                    #debug,
                    #DigTwin
                   ])

st.set_page_config(page_title="Eagle Eye", 
                   page_icon="🤖",
                   layout="wide")

st.sidebar.text("Powered By:")
# Footer di Sidebar menggunakan HTML dan CSS
footer = """
<style>
.sidebar .sidebar-content {
    padding-bottom: 50px;
}
.footer {
    position: relative;
    bottom: 0;
    width: 100%;
    text-align: left;
    padding: 10px;
    font-size: 12px;
    color: #555;
}

.footer img {
    margin: 0 5px 0 5px;
}

</style>
<div class="footer">
    <img src="https://dti.itb.ac.id/wp-content/uploads/2020/09/logo_itb_1024.png" alt="ITB" width="60px" height="auto">
</div>
<a href="https://github.com/aprianz042/people_crowd" target="_blank" style="text-decoration: none;">
    <span style="vertical-align: middle;">GitHub Repository</span>
</a>
"""

st.sidebar.markdown(footer, unsafe_allow_html=True)

pg.run()
