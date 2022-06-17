from multipage import MultiPage
from pages import about, predict

# Create an instance of the app 
app = MultiPage()

# Add all your applications (pages) here
app.add_page("About project", about.app)
app.add_page("Prediction", predict.app)

# The main app
app.run()