from explainerdashboard import ExplainerDashboard

# Загружаем дашборд из файла
db = ExplainerDashboard.from_config("abalone_regression_dashboard.yaml")

if __name__ == "__main__":
    db.run(host='0.0.0.0', port=8050)