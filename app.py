#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# Install web framework and redis
pip install Flask redis

# Install Flask-Dance (used for AWS SSO)
pip install Flask-Dance

# Replace the placeholders with your actual values. 
# You will need to create a Cognito User Pool and an App Client to obtain the client_id, client_secret, and region.
# Update your login route in app.py to authenticate with AWS SSO:
from flask import Flask, render_template, request, redirect, url_for
from flask_dance.contrib.cognito import make_cognito_blueprint, cognito

app = Flask(__name__)
app.secret_key = "your_super_secret_key"

cognito_blueprint = make_cognito_blueprint(
    client_id="your_cognito_client_id",
    client_secret="your_cognito_client_secret",
    region="your_aws_region",
    scope=["openid", "profile", "aws.cognito.signin.user.admin"],
    redirect_url="http://localhost:5000/login",
)
app.register_blueprint(cognito_blueprint, url_prefix="/login")

# Create a login route
@app.route('/login')
def login():
    if not cognito.authorized:
        return redirect(url_for("cognito.login"))
    resp = cognito.get("/oauth2/userInfo")
    assert resp.ok
    return "You are logged in with AWS SSO. Userinfo: {}".format(resp.json())

# Create a logout route
@app.route('/logout')
def logout():
    token = cognito_blueprint.token["access_token"]
    cognito_blueprint.session.get("/logout?client_id={}&logout_uri=http://localhost:5000".format(app.config["COGNITO_OAUTH_CLIENT_ID"]))
    cognito_blueprint.token = None
    return redirect(url_for('login'))

# Protect the chat route to require authentication
from flask_dance.consumer import oauth_authorized
from flask_login import LoginManager, UserMixin, current_user, login_required

login_manager = LoginManager(app)

class User(UserMixin):
    def __init__(self, id):
        self.id = id

@login_manager.user_loader
def load_user(user_id):
    return User(user_id)

@oauth_authorized.connect_via(cognito_blueprint)
def cognito_logged_in(blueprint, token):
    resp = blueprint.session.get("/oauth2/userInfo")
    user_info = resp.json()
    user_id = str(user_info["sub"])
    user = User(user_id)
    login_user(user)

@app.route('/chat', methods=['GET', 'POST'])
@login_required
def chat():
    if request.method == "POST":
        prompt = request.form["prompt"]
        # Add prompt to Redis cache
        redis_client.lpush("prompt_history", prompt)

        # Generate response using the LLM model  
        response = generate_text(prompt)

        return render_template("chat.html", response=response, prompt_history=redis_client.lrange("prompt_history", 0, -1))

    return render_template("chat.html", prompt_history=redis_client.lrange("prompt_history", 0, -1))

# Initialize Flask application and Redis client
from flask import Flask, render_template, request, redirect, url_for, flash
import redis

app = Flask(__name__)
app.secret_key = "your_secret_key"
redis_client = redis.Redis(host="localhost", port=6379, db=0)

# Add routes for login, logout, and chat
@app.route("/")
def home():
    return render_template("login.html")

@app.route("/chat", methods=["GET", "POST"])
def chat():
    if request.method == "POST":
        prompt = request.form["prompt"]
        # Add prompt to Redis cache
        redis_client.lpush("prompt_history", prompt)

        # Generate response using the LLM model  
        response = generate_text(prompt)

        return render_template("chat.html", response=response, prompt_history=redis_client.lrange("prompt_history", 0, -1))

    return render_template("chat.html", prompt_history=redis_client.lrange("prompt_history", 0, -1))

@app.route("/clear_history")
def clear_history():
    redis_client.delete("prompt_history")
    return redirect(url_for("chat"))

@app.route("/logout")
def logout():
    return redirect(url_for("home"))

# Add the following code at the end of app.py to run the Flask application:
if __name__ == "__main__":
    app.run(debug=True)

