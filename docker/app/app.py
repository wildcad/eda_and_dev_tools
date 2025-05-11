from explainerdashboard import ExplainerDashboard

db = ExplainerDashboard.from_config("abalone_regression_dashboard.yaml")
db.run(host='0.0.0.0', port=8050, use_waitress=True)