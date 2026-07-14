"""Development entry point for the Pediatric Malnutrition Detection web app."""

from src.web import create_app


app = create_app()


if __name__ == "__main__":
    app.run(debug=app.config["DEBUG"])
